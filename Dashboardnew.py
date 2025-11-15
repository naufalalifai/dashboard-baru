import base64
import os
import pickle
from functools import lru_cache
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------
# CONFIG – paths + feature list
# ---------------------------------------
DATA_PATH = "epl_clean1.csv"
MODEL_PATH = "team_form_model.pkl"
LOGO_DIR = Path("Logo")

FEATURE_COLS = [
    'is_home', 'season_week', 'days_since_last_match', 'h2h_win_rate',
    'avg_goals_scored_last5', 'avg_goals_conceded_last5',
    'avg_shots_last5', 'avg_shots_on_target_last5',
    'avg_shot_conversion_rate_last5', 'avg_shot_accuracy_rate_last5',
    'season_avg_goals_scored', 'season_avg_goals_conceded',
    'avg_shots_season', 'avg_shots_on_target_season',
    'avg_shot_conversion_rate_season', 'avg_shot_accuracy_rate_season',
    'previous_avg_goals_scored', 'previous_avg_goals_conceded',
    'previous_avg_shots', 'previous_avg_shots_on_target',
    'previous_avg_shot_conversion_rate', 'previous_avg_shot_accuracy_rate',
    'opp_avg_goals_scored_last5', 'opp_avg_goals_conceded_last5',
    'opp_avg_shots_last5', 'opp_avg_shots_on_target_last5',
    'opp_clean_sheet_rate_last5',
    'opp_avg_shot_conversion_rate_last5', 'opp_avg_shot_accuracy_rate_last5',
]


# ---------------------------------------
# LOADING + PREDICTIONS
# ---------------------------------------
@st.cache_data
def load_data_with_predictions(data_path: str, model_path: str) -> pd.DataFrame:
    """Load dataset, model, and add prediction columns for all rows."""
    df = pd.read_csv(data_path, parse_dates=["MatchDate"])
    df = df.reset_index(drop=True)

    # Ensure numeric features
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict for all rows
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    proba = model.predict_proba(X)[:, 1]
    pred_label = (proba >= 0.5).astype(int)

    df["PredProba"] = proba
    df["PredLabel"] = pred_label

    return df


# ---------------------------------------
# GENERIC HELPERS
# ---------------------------------------
def result_to_code(res):
    """Normalize result to 'W', 'D', or 'L'."""
    if pd.isna(res):
        return ""
    s = str(res).strip().upper()
    if s in ["W", "WIN", "WON"]:
        return "W"
    if s in ["D", "DRAW"]:
        return "D"
    if s in ["L", "LOSS", "LOST"]:
        return "L"
    return s[0]


def points_from_result(res):
    """Map result to league points."""
    code = result_to_code(res)
    if code == "W":
        return 3
    if code == "D":
        return 1
    return 0


def last5_record_string(df: pd.DataFrame, season, team, current_date):
    """
    Return W/L/D string of last 5 matches BEFORE current_date for this team & season,
    e.g. 'W D L L W'.
    """
    mask = (
        (df["Season"] == season) &
        (df["Team"] == team) &
        (df["MatchDate"] < current_date)
    )
    prev_matches = df[mask].sort_values("MatchDate", ascending=False)
    last5_raw = prev_matches["Result"].head(5).tolist()
    last5_codes = [result_to_code(r) for r in last5_raw]
    return " ".join(last5_codes[::-1])  # oldest → newest


def get_seasons(df: pd.DataFrame):
    return sorted(df["Season"].unique())


def get_teams_for_season(df: pd.DataFrame, season):
    return sorted(df.loc[df["Season"] == season, "Team"].unique())


def get_prev_season(seasons, current_season):
    seasons_sorted = sorted(seasons)
    if current_season not in seasons_sorted:
        return None
    idx = seasons_sorted.index(current_season)
    if idx == 0:
        return None
    return seasons_sorted[idx - 1]


def get_match_pair(df: pd.DataFrame, row_home: pd.Series):
    """
    Given a home-team row, return (home_row, away_row) for that match.
    Uses Season + MatchDate + swapped Team/Opponent.
    """
    season_val = row_home["Season"]
    date_val = row_home["MatchDate"]
    home = row_home["Team"]
    away = row_home["Opponent"]

    mask = (
        (df["Season"] == season_val) &
        (df["MatchDate"] == date_val) &
        (df["Team"] == away) &
        (df["Opponent"] == home)
    )
    opp_rows = df[mask]
    if opp_rows.empty:
        return row_home, None
    return row_home, opp_rows.iloc[0]


def calculate_time_weight(matchweek: int, total_matchweeks: int = 38) -> float:
    """
    Calculate time weight for stakes adjustment based on season progression.

    Season phases:
    - Early Season (0-25%): 10% weight (matchweeks 1-9)
    - Mid Season (25-65%): 40% weight (matchweeks 10-25)
    - Late Season (65-90%): 70% weight (matchweeks 26-34)
    - Final Stretch (90-100%): 100% weight (matchweeks 35-38)

    Args:
        matchweek: Current matchweek (1-38)
        total_matchweeks: Total matchweeks in season (default 38 for EPL)

    Returns:
        float: Time weight multiplier (0.10 - 1.00)

    Example:
        >>> calculate_time_weight(5)   # Early season
        0.10
        >>> calculate_time_weight(15)  # Mid season
        0.40
        >>> calculate_time_weight(30)  # Late season
        0.70
        >>> calculate_time_weight(37)  # Final stretch
        1.00
    """
    if matchweek <= (total_matchweeks * 0.25):  # First 25% (~9 games)
        return 0.30
    elif matchweek <= (total_matchweeks * 0.65):  # Up to 65% (~25 games)
        return 0.60
    elif matchweek <= (total_matchweeks * 0.90):  # Up to 90% (~34 games)
        return 0.90
    else:  # Final 10% (~35-38 games)
        return 1.00


def calculate_stakes_score(team_a_position: int, team_b_position: int, total_teams: int = 20, matchweek: int = 38) -> float:
    """
    Calculate stakes score (0-2.0) based on league positions and match context, adjusted by season timing.

    Args:
        team_a_position: League position of team A (1 = first place)
        team_b_position: League position of team B (1 = first place)
        total_teams: Total teams in league (default 20 for EPL)
        matchweek: Current matchweek (1-38, default 38 for full weight)

    Returns:
        float: Time-adjusted stakes score from 0.0 to 2.0

    Base scoring rules (priority order):
        - Title race (both top 3): 2.0
        - Relegation six-pointer (both bottom 3): 2.0
        - Relegation danger (both bottom 5): 1.7
        - Champions League race (both 4-7): 1.6
        - Title vs challenger (one top 3, other 4-6): 1.5
        - Europa League race (both 5-10): 1.2
        - David vs Goliath (gap >= 10): 0.5
        - Mid-table clash (both 8-15): 0.2
        - Default: 0.0

    Time adjustment:
        - Early season (weeks 1-9): 10% weight
        - Mid season (weeks 10-25): 40% weight
        - Late season (weeks 26-34): 70% weight
        - Final stretch (weeks 35-38): 100% weight
    """
    # Handle None/invalid positions
    if team_a_position is None or team_b_position is None:
        return 0.5  # Neutral stakes if position unknown

    # Calculate base stakes
    base_stakes = 0.0

    # Title race - both in top 3
    if team_a_position <= 3 and team_b_position <= 3:
        base_stakes = 2.0

    # Relegation six-pointer - both in bottom 3
    elif team_a_position >= (total_teams - 2) and team_b_position >= (total_teams - 2):
        base_stakes = 2.0

    # Relegation danger - both in bottom 5
    elif team_a_position >= (total_teams - 4) and team_b_position >= (total_teams - 4):
        base_stakes = 1.7

    # Champions League race - both in positions 4-7
    elif (4 <= team_a_position <= 7) and (4 <= team_b_position <= 7):
        base_stakes = 1.6

    # Title contender vs challenger
    elif (team_a_position <= 3 and 4 <= team_b_position <= 6) or \
         (team_b_position <= 3 and 4 <= team_a_position <= 6):
        base_stakes = 1.5

    # Europa League race - both in positions 5-10
    elif (5 <= team_a_position <= 10) and (5 <= team_b_position <= 10):
        base_stakes = 1.2

    # David vs Goliath - large position gap
    elif abs(team_a_position - team_b_position) >= 10:
        base_stakes = 0.5

    # Mid-table clash - both in positions 8-15
    elif (8 <= team_a_position <= 15) and (8 <= team_b_position <= 15):
        base_stakes = 0.2

    # Apply time weight
    time_weight = calculate_time_weight(matchweek)
    adjusted_stakes = base_stakes * time_weight

    return adjusted_stakes


def calculate_match_worthiness(
    team_a_prob: float,
    team_b_prob: float,
    team_a_position: int = None,
    team_b_position: int = None,
    total_teams: int = 20,
    matchweek: int = 38
) -> dict:
    """
    Calculate comprehensive match worthiness score (0-10 scale) with time-adjusted stakes.

    Combines 4 factors:
    - Quality (40%): Average team performance level
    - Competitiveness (25%): How evenly matched teams are
    - Unpredictability (15%): Outcome uncertainty
    - Stakes (20%): Match importance based on league position and season timing

    Args:
        team_a_prob: Team A performance probability (0.0-1.0)
        team_b_prob: Team B performance probability (0.0-1.0)
        team_a_position: Team A league position (optional)
        team_b_position: Team B league position (optional)
        total_teams: Total teams in league (default 20)
        matchweek: Current matchweek (1-38, default 38 for full weight)

    Returns:
        dict with keys:
            - total_score (0-10)
            - quality_score (0-4)
            - competitiveness_score (0-2.5)
            - unpredictability_score (0-1.5)
            - stakes_score (0-2.0, time-adjusted)
            - matchweek (int)
            - time_weight (float)
            - recommendation (classification label)
            - priority (priority level)
            - breakdown (dict with component percentages)
    """
    # Factor 1: Quality (0-4 points, 40%)
    avg_performance = (team_a_prob + team_b_prob) / 2
    quality_score = avg_performance * 4

    # Factor 2: Competitiveness (0-2.5 points, 25%)
    prob_difference = abs(team_a_prob - team_b_prob)
    competitiveness_score = (1 - prob_difference) * 2.5

    # Factor 3: Unpredictability (0-1.5 points, 15%)
    # Probabilities near 0.5 = most unpredictable
    unpred_a = max(0, 1 - abs(team_a_prob - 0.5) * 2)
    unpred_b = max(0, 1 - abs(team_b_prob - 0.5) * 2)
    avg_unpredictability = (unpred_a + unpred_b) / 2
    unpredictability_score = avg_unpredictability * 1.5

    # Factor 4: Stakes with time adjustment (0-2.0 points, 20%)
    stakes_score = calculate_stakes_score(team_a_position, team_b_position, total_teams, matchweek)

    # Total score
    total_score = quality_score + competitiveness_score + unpredictability_score + stakes_score

    # Classification
    recommendation, priority = classify_match_score(total_score)

    # Calculate breakdown percentages
    breakdown = {
        "quality_pct": round((quality_score / total_score * 100) if total_score > 0 else 0, 2),
        "competitiveness_pct": round((competitiveness_score / total_score * 100) if total_score > 0 else 0, 2),
        "unpredictability_pct": round((unpredictability_score / total_score * 100) if total_score > 0 else 0, 2),
        "stakes_pct": round((stakes_score / total_score * 100) if total_score > 0 else 0, 2),
    }

    return {
        "total_score": round(total_score, 2),
        "quality_score": round(quality_score, 2),
        "competitiveness_score": round(competitiveness_score, 2),
        "unpredictability_score": round(unpredictability_score, 2),
        "stakes_score": round(stakes_score, 2),
        "matchweek": matchweek,
        "time_weight": calculate_time_weight(matchweek),
        "recommendation": recommendation,
        "priority": priority,
        "breakdown": breakdown
    }


def classify_match_score(total_score: float) -> tuple:
    """
    Classify match worthiness score into recommendation category.

    Args:
        total_score: Match worthiness score (0-10)

    Returns:
        tuple: (recommendation_text, priority_level)

    Categories:
        - 5.0-10.0: Worth Watching (WORTH)
        - 3.0-4.9: Maybe (MAYBE)
        - 0.0-2.9: Skip (SKIP)
    """
    if total_score >= 5.0:
        return "👍 Worth Watching", "WORTH"
    elif total_score >= 3.5:
        return "🤔 Maybe", "MAYBE"
    else:
        return "⏭️ Skip", "SKIP"


def build_full_league_table(df: pd.DataFrame, season):
    """
    Full-season league table for a given season.
    """
    season_df = df[df["Season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame()

    season_df["ResultCode"] = season_df["Result"].apply(result_to_code)
    season_df["Points"] = season_df["ResultCode"].apply(points_from_result)

    grouped = season_df.groupby("Team").agg(
        Played=("ResultCode", "count"),
        Wins=("ResultCode", lambda x: (x == "W").sum()),
        Draws=("ResultCode", lambda x: (x == "D").sum()),
        Losses=("ResultCode", lambda x: (x == "L").sum()),
        GoalsFor=("GoalsFor", "sum"),
        GoalsAgainst=("GoalsAgainst", "sum"),
        Points=("Points", "sum"),
    ).reset_index()

    grouped["GoalDiff"] = grouped["GoalsFor"] - grouped["GoalsAgainst"]
    grouped = grouped.sort_values(
        ["Points", "GoalDiff", "GoalsFor"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    grouped.index = grouped.index + 1
    grouped.insert(0, "Pos", grouped.index)
    return grouped


def build_league_table_up_to_week(df: pd.DataFrame, season, current_week):
    """
    League table up to a given gameweek (1..N).
    """
    season_df = df[(df["Season"] == season) & (df["season_week"] <= current_week)].copy()
    if season_df.empty:
        return pd.DataFrame()

    season_df["ResultCode"] = season_df["Result"].apply(result_to_code)
    season_df["Points"] = season_df["ResultCode"].apply(points_from_result)

    grouped = season_df.groupby("Team").agg(
        Played=("ResultCode", "count"),
        Wins=("ResultCode", lambda x: (x == "W").sum()),
        Draws=("ResultCode", lambda x: (x == "D").sum()),
        Losses=("ResultCode", lambda x: (x == "L").sum()),
        GoalsFor=("GoalsFor", "sum"),
        GoalsAgainst=("GoalsAgainst", "sum"),
        Points=("Points", "sum"),
    ).reset_index()

    grouped["GoalDiff"] = grouped["GoalsFor"] - grouped["GoalsAgainst"]
    grouped = grouped.sort_values(
        ["Points", "GoalDiff", "GoalsFor"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    grouped.index = grouped.index + 1
    grouped.insert(0, "Pos", grouped.index)
    return grouped


def build_position_trends(df: pd.DataFrame, season: str, max_week: int) -> pd.DataFrame:
    """Build cumulative position trajectory for each club up to max_week."""
    if max_week <= 0:
        return pd.DataFrame()

    frames = []
    for week in range(1, max_week + 1):
        week_table = build_league_table_up_to_week(df, season, week)
        if week_table.empty:
            continue
        frame = week_table[["Team", "Pos"]].copy()
        frame["Week"] = week
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_team_position_at_week(df: pd.DataFrame, season, team: str, week: int) -> int:
    """
    Get a team's league position at a specific gameweek.

    Args:
        df: Full dataset
        season: Season identifier
        team: Team name
        week: Gameweek number (uses week-1 for completed matches)

    Returns:
        int: League position (1-20), or None if not found
    """
    # Use week-1 to get position based on completed matches
    # For gameweek 1, position is unknown (no matches played)
    if week <= 1:
        return None

    # Build table up to previous week (completed matches only)
    table = build_league_table_up_to_week(df, season, week - 1)

    if table.empty:
        return None

    # Find team's position
    team_row = table[table["Team"] == team]
    if team_row.empty:
        return None

    return int(team_row.iloc[0]["Pos"])


def get_stakes_context(team_a_pos: int, team_b_pos: int, total_teams: int = 20) -> str:
    """
    Get descriptive context label for match stakes.

    Returns:
        str: Context label (e.g., "Title Race", "Relegation Battle")
    """
    if team_a_pos is None or team_b_pos is None:
        return "Unknown Stakes"

    # Same logic as calculate_stakes_score
    if team_a_pos <= 3 and team_b_pos <= 3:
        return "Title Race"

    relegation_zone = total_teams - 2
    if team_a_pos >= relegation_zone and team_b_pos >= relegation_zone:
        return "Relegation Six-Pointer"

    danger_zone = total_teams - 4
    if team_a_pos >= danger_zone and team_b_pos >= danger_zone:
        return "Relegation Battle"

    if (4 <= team_a_pos <= 7) and (4 <= team_b_pos <= 7):
        return "Champions League Race"

    if (team_a_pos <= 3 and 4 <= team_b_pos <= 6) or \
       (team_b_pos <= 3 and 4 <= team_a_pos <= 6):
        return "Title Contender vs Challenger"

    if (5 <= team_a_pos <= 10) and (5 <= team_b_pos <= 10):
        return "Europa League Race"

    if abs(team_a_pos - team_b_pos) >= 10:
        return "David vs Goliath"

    if (8 <= team_a_pos <= 15) and (8 <= team_b_pos <= 15):
        return "Mid-Table Clash"

    return "No Significant Stakes"


def to_display_series(row: pd.Series):
    """Copy row and replace numeric NaNs with 0 for display."""
    df_temp = row.to_frame().T
    num_cols = df_temp.select_dtypes(include=["number"]).columns
    df_temp[num_cols] = df_temp[num_cols].fillna(0)
    return df_temp.iloc[0]


def compute_team_ratings(df_raw: pd.DataFrame, current_season, team: str):
    """
    Compute 1–5 star ratings (overall, attack, defense, control) for a team,
    based on:
      - first season in dataset  -> global averages across ALL seasons
      - later seasons            -> previous season's averages only

    Metrics:
      - Overall : MPI
      - Attack  : Off_raw_norm
      - Defense : Def_raw_norm
      - Control : Ctrl_raw_norm
    """
    # Global per-team
    global_agg = (
        df_raw.groupby("Team")
        .agg({
            "MPI": "mean",
            "Off_raw_norm": "mean",
            "Def_raw_norm": "mean",
            "Ctrl_raw_norm": "mean",
        })
        .reset_index()
    )

    season_team_agg = (
        df_raw.groupby(["Season", "Team"])
        .agg({
            "MPI": "mean",
            "Off_raw_norm": "mean",
            "Def_raw_norm": "mean",
            "Ctrl_raw_norm": "mean",
        })
        .reset_index()
    )

    seasons_sorted = sorted(df_raw["Season"].unique())
    if not seasons_sorted:
        return {"overall": 3, "attack": 3, "defense": 3, "control": 3,
                "source": "none", "prev_season": None}

    earliest = seasons_sorted[0]

    if current_season not in seasons_sorted:
        base_agg = global_agg
        row = base_agg[base_agg["Team"] == team]
        source = "global"
        prev_season = None
    elif current_season == earliest:
        base_agg = global_agg
        row = base_agg[base_agg["Team"] == team]
        source = "global"
        prev_season = None
    else:
        idx = seasons_sorted.index(current_season)
        prev_season = seasons_sorted[idx - 1]
        season_agg = season_team_agg[season_team_agg["Season"] == prev_season]
        row = season_agg[season_agg["Team"] == team]
        if row.empty:
            base_agg = global_agg
            row = base_agg[base_agg["Team"] == team]
            source = "global"
            prev_season = None
        else:
            base_agg = season_agg
            source = "prev"

    if row.empty:
        return {"overall": 3, "attack": 3, "defense": 3, "control": 3,
                "source": source, "prev_season": prev_season}

    row = row.iloc[0]

    def scale_to_stars(val, series, reverse=False):
        series = series.dropna()
        if series.empty or pd.isna(val):
            return 3
        vmin, vmax = series.min(), series.max()
        if vmin == vmax:
            return 3
        if reverse:
            norm = (vmax - val) / (vmax - vmin)
        else:
            norm = (val - vmin) / (vmax - vmin)
        stars = 1 + norm * 4
        stars = int(round(stars))
        stars = max(1, min(5, stars))
        return stars

    overall = scale_to_stars(row["MPI"], base_agg["MPI"], reverse=False)
    attack = scale_to_stars(row["Off_raw_norm"], base_agg["Off_raw_norm"], reverse=False)
    defense = scale_to_stars(row["Def_raw_norm"], base_agg["Def_raw_norm"], reverse=False)
    control = scale_to_stars(row["Ctrl_raw_norm"], base_agg["Ctrl_raw_norm"], reverse=False)

    return {
        "overall": overall,
        "attack": attack,
        "defense": defense,
        "control": control,
        "source": source,
        "prev_season": prev_season,
    }


def _normalize_team_name(name: str) -> str:
    """Remove non-alphanumeric characters and lowercase for matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _logo_lookup() -> dict:
    """Build a mapping of normalized team name -> logo path."""
    mapping = {}
    if LOGO_DIR.exists():
        for file in LOGO_DIR.glob("*.png"):
            mapping[_normalize_team_name(file.stem)] = str(file)
    return mapping


def to_base64(path: str) -> str:
    """Return file contents encoded as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_logo_path(team: str):
    """
    Try to load a logo from the Logo directory, matching even if the filename
    uses spaces, apostrophes, or other punctuation.
    """
    if not team:
        return None

    # Exact filename match first (e.g., "Man City.png")
    direct_path = LOGO_DIR / f"{team}.png"
    if direct_path.exists():
        return str(direct_path)

    normalized = _normalize_team_name(team)
    if not normalized:
        return None

    return _logo_lookup().get(normalized)


# ---------------------------------------
# PAGE 0 – User Guide
# ---------------------------------------
def page_user_guide():
    st.title("📖 Quick Start Guide")

    st.markdown("""
    Welcome! This dashboard helps you find the **most exciting EPL matches to watch** based on team form, match importance, and entertainment value.
    """)

    # Quick Start Section
    st.markdown("---")
    st.markdown("## 🚀 Get Started in 3 Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Setup")
        st.markdown("""
        Go to **Select Season & Club**
        - Pick your season
        - Select favourite clubs (optional)
        """)

    with col2:
        st.markdown("### 2️⃣ Explore")
        st.markdown("""
        Visit **Match Watchability**
        - Browse gameweeks (use ◀ ▶)
        - Check match scores (0-10)
        - See recommendations
        """)

    with col3:
        st.markdown("### 3️⃣ Analyze")
        st.markdown("""
        Click the **View Match Result Analysis** button
        - See predictions vs results
        - Review team stats
        - Check accuracy
        """)

    # Pages Overview Section
    st.markdown("---")
    st.markdown("## 📑 What Each Page Does")

    st.markdown("### ⚽ Select Season & Club")
    st.markdown("""
    **Start here!** Choose your season and favourite teams.
    - Star ratings show team quality based on past performance (1-5 ⭐)
    - Settings apply to all pages
    """)

    st.markdown("### 📺 Match Watchability")
    st.markdown("""
    **Find exciting matches!** Browse gameweeks and see match scores (0-10).
    - 👍 **5.0+** = Worth Watching
    - 🤔 **3.5-4.9** = Maybe
    - ⏭️ **<3.5** = Skip
    - Match score considers: Team quality (40%), how evenly matched (25%), outcome uncertainty (15%), and match importance (20%)
    """)

    st.markdown("### 📋 Match Result Analysis")
    st.markdown("""
    **Check accuracy!** See if predictions matched reality.
    - Access via the **View Match Result Analysis** button (on Match Watchability)
    - Compare predicted vs actual form
    - View detailed match stats
    """)

    st.markdown("### 📈 Club Stats")
    st.markdown("""
    **Club deep-dive!** Season stats for any team.
    - Win/draw/loss records
    - Form predictions accuracy
    - Match-by-match history
    """)

    st.markdown("### 🏆 League Standings")
    st.markdown("""
    **Track standings!** See league positions over time.
    - League table showing completed matches only
    - Position trend charts
    - Understand match stakes context
    """)

    # Key Features Section
    st.markdown("---")
    st.markdown("## ✨ Understanding the Scores")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Match Score (0-10)
        Combines 4 factors:
        - **Quality:** Team performance level
        - **Competitiveness:** How evenly matched
        - **Unpredictability:** Outcome uncertainty
        - **Stakes:** Match importance

        Higher score = More exciting to watch!
        """)

        st.markdown("""
        ### 🎯 Form Predictions
        AI predicts if teams are "on-form" or "off-form" using:
        - Recent results (last 5 games)
        - Goals & shots statistics
        - Head-to-head history

        **Confidence** shown as % (e.g., 75.3%)
        """)

    with col2:
        st.markdown("""
        ### ⭐ Team Ratings (1-5 Stars)
        - **Overall:** Match Performance Index
        - **Attack:** Goal-scoring ability
        - **Defense:** Defensive strength
        - **Control:** Possession & game management
        """)

        st.markdown("""
        ### 📅 Stakes Timing
        Match importance grows as the season progresses:
        - Early (Weeks 1-9): 30% weight (lower stakes)
        - Mid (Weeks 10-25): 60% weight (moderate stakes)
        - Late (Weeks 26-34): 90% weight (high stakes)
        - Final (Weeks 35-38): 100% weight (critical stakes)

        Same fixture has higher stakes later in the season!
        """)

    # Tips & Best Practices
    st.markdown("---")
    st.markdown("## 💡 Quick Tips")

    st.markdown("""
    ### Navigation
    1. **Start at Select Season & Club** - Choose your preferences first
    2. **Use ◀ ▶ buttons** - Browse gameweeks quickly
    3. **Filter by favourites** - Focus on teams you care about
    4. **Click the View Match Result Analysis button** - Jump to detailed stats

    ### Finding Great Matches
    - **5.0+ score** = Must watch
    - **High stakes** = Important matches (title race, relegation)
    - **Close team positions** = Competitive games

    ### Good to Know
    - League table shows completed matches only
    - Stakes matter more late in the season
    - Predictions aren't perfect - check accuracy on the Match Result Analysis page.
    """)

    # Glossary Section
    st.markdown("---")
    st.markdown("## 📚 Key Terms")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **On-Form** - Team performing well recently

        **Off-Form** - Team underperforming recently

        **MPI** - Match Performance Index: Overall team performance score

        **H2H** - Head-to-head win rate: Past results when teams played each other
        """)

    with col2:
        st.markdown("""
        **GW** - Gameweek (1 round of matches)

        **Pos** - League position (1 = first, 20 = last)

        **GD** - Goal Difference: Goals scored minus goals conceded

        **Stakes** - Match importance: Title race, relegation battle, European qualification, etc.
        """)

    # Footer with call to action
    st.markdown("---")
    st.success("🎉 Ready to start! Go to **Select Season & Club** to begin.")
    st.caption("💡 Tip: You can return to this guide anytime from the sidebar menu.")


# ---------------------------------------
# PAGE 1 – Select Season & Club
# ---------------------------------------
def page_setup(df: pd.DataFrame):
    st.title("⚽ Match Watchability – Select Season & Club")
    st.info("Configure your season and favorite clubs here. Your selections will apply to all pages.")

    seasons = get_seasons(df)
    if not seasons:
        st.error("No seasons found in dataset.")
        return

    # Initialize session state for selected_season if not exists
    if "selected_season" not in st.session_state:
        st.session_state.selected_season = seasons[0]

    # Select season
    default_season_idx = seasons.index(st.session_state.selected_season) if st.session_state.selected_season in seasons else 0
    selected_season = st.selectbox(
        "Select season", seasons, index=default_season_idx, key="season_select"
    )

    # Update session state only if changed
    if st.session_state.selected_season != selected_season:
        st.session_state.selected_season = selected_season
        # Reset gameweek when season changes
        if "selected_gameweek" in st.session_state:
            del st.session_state.selected_gameweek
        # Reset selected match when season changes
        if "selected_match" in st.session_state:
            del st.session_state.selected_match

    teams = get_teams_for_season(df, selected_season)
    st.write(f"Teams in {selected_season}: {len(teams)}")

    # Initialize fav_clubs in session state if not exists
    if "fav_clubs" not in st.session_state:
        st.session_state.fav_clubs = []

    # Multi-select favourite clubs (optional)
    fav_clubs = st.multiselect(
        "Select your favourite clubs (optional – for filtering in Dashboard):",
        options=teams,
        default=st.session_state.fav_clubs if all(club in teams for club in st.session_state.fav_clubs) else [],
        key="fav_clubs_select",
    )
    st.session_state.fav_clubs = fav_clubs

    # Previous season league table
    prev_season = get_prev_season(seasons, selected_season)
    if prev_season:
        prev_table = build_full_league_table(df, prev_season)
        prev_pos_map = dict(zip(prev_table["Team"], prev_table["Pos"]))
    else:
        prev_table = None
        prev_pos_map = {}

    st.markdown("### Clubs Overview")

    rows = []
    for team in teams:
        ratings = compute_team_ratings(df, selected_season, team)
        prev_pos = prev_pos_map.get(team, None)
        rows.append({
            "Team": team,
            "Prev Season Pos" if prev_season else "Prev Season Pos (N/A)": prev_pos,
            "Overall ⭐": ratings["overall"],
            "Attack ⚽": ratings["attack"],
            "Defense 🛡️": ratings["defense"],
            "Control 🎮": ratings["control"],
        })

    overview_df = pd.DataFrame(rows)
    st.dataframe(overview_df, use_container_width=True)

    if prev_season:
        st.caption(
            f"Previous season: {prev_season}. Positions are based on full-season standings. "
            "Ratings are calculated using global or previous season performance data."
        )
    else:
        st.caption(
            "This is the earliest season in the dataset. Ratings are calculated using global historical performance data."
        )

    st.success(
        "✅ Filters saved! You can now navigate to the **Match Watchability** page to view fixtures."
    )


# ---------------------------------------
# PAGE 2 – Match Watchability
# ---------------------------------------
def page_watchability(df: pd.DataFrame):
    st.title("📺 Match Watchability")

    if "selected_season" not in st.session_state or st.session_state.selected_season is None:
        st.warning("Please go to **Select Season & Club** first to choose a season.")
        return

    season = st.session_state.selected_season
    fav_clubs = st.session_state.get("fav_clubs", [])

    df_season = df[df["Season"] == season].copy()
    if df_season.empty:
        st.error(f"No data for season {season}.")
        return

    min_week = int(df_season["season_week"].min())
    max_week = int(df_season["season_week"].max())

    # Initialize selected_gameweek in session state if not exists
    if "selected_gameweek" not in st.session_state:
        st.session_state.selected_gameweek = min_week

    # Initialize view_mode in session state if not exists
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "All fixtures"

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        # Clamp current week into available range
        if st.session_state.selected_gameweek < min_week:
            st.session_state.selected_gameweek = min_week
        if st.session_state.selected_gameweek > max_week:
            st.session_state.selected_gameweek = max_week

        st.markdown("##### Gameweek")
        controls = st.columns([1.5, 2, 1.5])
        prev_disabled = st.session_state.selected_gameweek <= min_week
        next_disabled = st.session_state.selected_gameweek >= max_week

        with controls[0]:
            st.write("")
            prev_clicked = st.button("◀", key="gw_minus", use_container_width=True, type="secondary", disabled=prev_disabled)

        with controls[2]:
            st.write("")
            next_clicked = st.button("▶", key="gw_plus", use_container_width=True, type="secondary", disabled=next_disabled)

        if prev_clicked:
            st.session_state.selected_gameweek = max(min_week, st.session_state.selected_gameweek - 1)
        if next_clicked:
            st.session_state.selected_gameweek = min(max_week, st.session_state.selected_gameweek + 1)

        current_week = st.session_state.selected_gameweek

        with controls[1]:
            st.markdown(
                f"<div style='text-align:center;font-size:2.5rem;font-weight:700;'>"
                f"GW {current_week}</div>",
                unsafe_allow_html=True,
            )

        gw = current_week
    with col_top2:
        mode = st.radio(
            "Which fixtures do you want to see?",
            ["All fixtures", "Favourite clubs only"],
            index=0 if st.session_state.view_mode == "All fixtures" else 1,
            horizontal=True,
        )
        # Update session state
        st.session_state.view_mode = mode

    week_df = df_season[df_season["season_week"] == gw]
    home_rows = week_df[week_df["is_home"] == 1].copy()

    if mode == "Favourite clubs only" and fav_clubs:
        # Keep matches where home OR away is in favourite clubs
        mask = home_rows["Team"].isin(fav_clubs) | home_rows["Opponent"].isin(fav_clubs)
        home_rows = home_rows[mask]

    if home_rows.empty:
        st.info("No matches found for this filter (possibly no favourite clubs playing this week).")
        return

    st.markdown(f"### Gameweek {gw} fixtures – {season}")

    # Build match summaries with new worthiness scoring
    match_summaries = []
    for idx, row_home in home_rows.iterrows():
        home_row, away_row = get_match_pair(df_season, row_home)
        if away_row is None:
            continue

        home_proba = float(home_row["PredProba"])
        away_proba = float(away_row["PredProba"])
        home_label = int(home_row["PredLabel"])
        away_label = int(away_row["PredLabel"])

        # Get league positions for stakes calculation
        home_team = home_row["Team"]
        away_team = home_row["Opponent"]
        home_position = get_team_position_at_week(df, season, home_team, gw)
        away_position = get_team_position_at_week(df, season, away_team, gw)

        # Calculate match worthiness score
        worthiness = calculate_match_worthiness(
            team_a_prob=home_proba,
            team_b_prob=away_proba,
            team_a_position=home_position,
            team_b_position=away_position,
            matchweek=gw
        )

        # Get stakes context
        stakes_context = get_stakes_context(home_position, away_position)

        match_summaries.append({
            "home_index": int(home_row.name),
            "away_index": int(away_row.name),
            "MatchDate": home_row["MatchDate"],
            "Home": home_row["Team"],
            "Away": home_row["Opponent"],
            "HomeProba": home_proba,
            "AwayProba": away_proba,
            "HomeLabel": home_label,
            "AwayLabel": away_label,
            "HomePosition": home_position,
            "AwayPosition": away_position,
            "Watchability": worthiness["recommendation"],  # New system
            "WorthinessScore": worthiness["total_score"],
            "QualityScore": worthiness["quality_score"],
            "CompetitivenessScore": worthiness["competitiveness_score"],
            "UnpredictabilityScore": worthiness["unpredictability_score"],
            "StakesScore": worthiness["stakes_score"],
            "Priority": worthiness["priority"],
            "StakesContext": stakes_context,
            "Breakdown": worthiness["breakdown"],
        })

    if not match_summaries:
        st.info("No valid home/away match pairs found for this gameweek.")
        return

    # Helper function to get color for score
    def get_score_color(score):
        if score >= 5.0:
            return "#28a745"  # Green - Worth Watching
        elif score >= 3.0:
            return "#ffa500"  # Orange - Maybe
        else:
            return "#dc3545"  # Red - Skip

    # Show summary table with new scoring
    summary_df = pd.DataFrame([
        {
            "Match": f"{m['Home']} vs {m['Away']}",
            "Date": m["MatchDate"].date(),
            "Score": f"{m['WorthinessScore']:.2f}/10",
            "Recommendation": m["Watchability"],
            "Stakes": m["StakesContext"],
            "Home (Pos)": f"{m['Home']} ({m['HomePosition'] if m['HomePosition'] else '-'})",
            "Away (Pos)": f"{m['Away']} ({m['AwayPosition'] if m['AwayPosition'] else '-'})",
        }
        for m in match_summaries
    ])

    # Display table
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.info("💡 **Tip:** First, select a match from the dropdown below. Then click **View Match Result Analysis** to see detailed statistics.")

    # Add score distribution summary
    st.markdown("#### Gameweek Summary")
    col1, col2, col3 = st.columns(3)

    worth_watching = sum(1 for m in match_summaries if m["Priority"] == "WORTH")
    maybe = sum(1 for m in match_summaries if m["Priority"] == "MAYBE")
    skip = sum(1 for m in match_summaries if m["Priority"] == "SKIP")
    
    col1.metric("👍 Worth Watching", worth_watching)
    col2.metric("🤔 Maybe", maybe)
    col3.metric("⏭️ Skip", skip)

    # Select a match for detailed view
    options = [f"{m['Home']} vs {m['Away']}" for m in match_summaries]
    selected_match = st.selectbox("Select a match to inspect in detail:", options)
    sel_idx = options.index(selected_match)
    sel = match_summaries[sel_idx]

    # Detailed view
    home_row = df_season.loc[sel["home_index"]]
    away_row = df_season.loc[sel["away_index"]]

    home_disp = to_display_series(home_row)
    away_disp = to_display_series(away_row)

    st.markdown("---")
    st.markdown(
        f"### Detailed view: **{sel['Home']}** vs **{sel['Away']}** "
        f"({sel['MatchDate'].date()}, GW {gw})"
    )

    # Worthiness Score Breakdown
    score_color = get_score_color(sel["WorthinessScore"])
    st.markdown(f"#### Match Watchability Score: <span style='color:{score_color}; font-size:2em; font-weight:bold;'>{sel['WorthinessScore']:.2f}/10</span>", unsafe_allow_html=True)
    st.markdown(f"**Recommendation:** {sel['Watchability']} (Priority: {sel['Priority']})")
    st.markdown(f"**Stakes Context:** {sel['StakesContext']}")

    # Score breakdown with progress bars
    st.markdown("##### Score Components")
    col_q, col_c, col_u, col_s = st.columns(4)

    with col_q:
        st.write("**Quality** (40%)")
        st.progress(sel["QualityScore"] / 4.0)
        st.caption(f"{sel['QualityScore']:.2f} / 4.0")

    with col_c:
        st.write("**Competitiveness** (25%)")
        st.progress(sel["CompetitivenessScore"] / 2.5)
        st.caption(f"{sel['CompetitivenessScore']:.2f} / 2.5")

    with col_u:
        st.write("**Unpredictability** (15%)")
        st.progress(sel["UnpredictabilityScore"] / 1.5)
        st.caption(f"{sel['UnpredictabilityScore']:.2f} / 1.5")

    with col_s:
        st.write("**Stakes** (20%)")
        st.progress(sel["StakesScore"] / 2.0)
        st.caption(f"{sel['StakesScore']:.2f} / 2.0")

    st.markdown("---")
    st.markdown("#### Team Form Analysis")

    c1, c2 = st.columns(2)
    with c1:
        pos_text = f" (Position: {sel['HomePosition']})" if sel['HomePosition'] else ""
        st.markdown(f"**{sel['Home']}{pos_text}**")
        arrow = "⬆️ On-form" if sel["HomeLabel"] == 1 else "⬇️ Off-form"
        st.metric("Predicted form", arrow, f"{sel['HomeProba']*100:.2f}%")
        st.write(f"Last 5 results: {last5_record_string(df, season, sel['Home'], sel['MatchDate'])}")
        st.write(f"Head-to-Head win rate vs {sel['Away']}: {home_disp['h2h_win_rate']*100:.2f}%")

    with c2:
        pos_text = f" (Position: {sel['AwayPosition']})" if sel['AwayPosition'] else ""
        st.markdown(f"**{sel['Away']}{pos_text}**")
        arrow_o = "⬆️ On-form" if sel["AwayLabel"] == 1 else "⬇️ Off-form"
        st.metric("Predicted form", arrow_o, f"{sel['AwayProba']*100:.2f}%")
        st.write(f"Last 5 results: {last5_record_string(df, season, sel['Away'], sel['MatchDate'])}")
        st.write(f"Head-to-Head win rate (from {sel['Away']} perspective): {away_disp['h2h_win_rate']*100:.2f}%")

    st.markdown("#### Key Historical Stats (Last 5 Matches & Season Averages)")

    stats_cols_last5 = [
        ("avg_goals_scored_last5", "Avg goals scored"),
        ("avg_goals_conceded_last5", "Avg goals conceded"),
        ("avg_shots_last5", "Avg shots"),
        ("avg_shots_on_target_last5", "Avg shots on target"),
        ("avg_shot_conversion_rate_last5", "Shot conversion rate"),
        ("avg_shot_accuracy_rate_last5", "Shot accuracy rate"),
    ]
    stats_cols_season = [
        ("season_avg_goals_scored", "Avg goals scored"),
        ("season_avg_goals_conceded", "Avg goals conceded"),
        ("avg_shots_season", "Avg shots"),
        ("avg_shots_on_target_season", "Avg shots on target"),
        ("avg_shot_conversion_rate_season", "Shot conversion rate"),
        ("avg_shot_accuracy_rate_season", "Shot accuracy rate"),
    ]

    home_label = sel["Home"]
    away_label = sel["Away"]

    def format_stat(col_name, value):
        if pd.isna(value):
            return "-"
        if "rate" in col_name or "accuracy" in col_name:
            return f"{float(value) * 100:.2f}%"
        return f"{float(value):.2f}"

    def build_comparison_rows(metric_config):
        rows = []
        for col, display_name in metric_config:
            rows.append({
                home_label: format_stat(col, home_disp[col]),
                "Metric": display_name,
                away_label: format_stat(col, away_disp[col]),
            })
        return rows

    last5_df = pd.DataFrame(build_comparison_rows(stats_cols_last5))
    season_df = pd.DataFrame(build_comparison_rows(stats_cols_season))

    st.markdown("##### Last 5 matches")
    st.dataframe(
        last5_df[[home_label, "Metric", away_label]],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### Season averages")
    st.dataframe(
        season_df[[home_label, "Metric", away_label]],
        use_container_width=True,
        hide_index=True,
    )

    # Button to navigate to Match Result View
    st.markdown("---")
    st.markdown("### 📋 View Match Result Analysis")
    st.write(f"See detailed analysis for **{sel['Home']} vs {sel['Away']}** including:")
    st.write("• Actual match results and final score")
    st.write("• Prediction accuracy (predicted vs actual form)")
    st.write("• Detailed match statistics (shots, corners, cards, fouls)")

    if st.button("📋 View Match Result Analysis", type="primary", use_container_width=True):
        # Store the selected match details in session state
        st.session_state.selected_match = {
            "home_index": sel["home_index"],
            "away_index": sel["away_index"],
            "Home": sel["Home"],
            "Away": sel["Away"],
            "MatchDate": sel["MatchDate"],
            "season": season,
            "gameweek": gw,
        }
        # Activate Match Result Analysis mode
        st.session_state.match_result_mode = True
        st.rerun()

    st.caption(
        "Use this page to scan gameweek fixtures and identify which matches are worth watching. "
        "Look for high-quality clashes, competitive games, or skip low-excitement matches."
    )


# ---------------------------------------
# PAGE 3 – Match Result Analysis
# ---------------------------------------
def page_match_result(df: pd.DataFrame):
    st.title("📋 Match Result Analysis")

    if "selected_season" not in st.session_state or st.session_state.selected_season is None:
        st.warning("Please go to **Select Season & Club** first to choose a season.")
        return

    # Check if a match has been selected from Dashboard
    if "selected_match" not in st.session_state or st.session_state.selected_match is None:
        st.info("📍 No match selected. Please select a match from the **Match Watchability** page first.")
        st.markdown("### How to Use This Page:")
        st.markdown("""
        This page provides detailed match analysis with prediction validation. To access it:

        1. **Navigate to Match Watchability** (from the sidebar)
        2. **Select a gameweek** using the ◀ ▶ buttons
        3. **Choose a match** from the "Select a match to inspect in detail" dropdown
        4. **Click the "📋 View Match Result Analysis" button** at the bottom of the match details

        The selected match will then appear here with comprehensive statistics and form predictions!
        """)

        # Add back button to return to dashboard
        if st.button("⬅️ Back to Match Watchability", type="primary"):
            st.session_state.match_result_mode = False
            st.session_state.current_page = "2. Match Watchability"
            st.rerun()
        return
    
    # Get selected match from session state
    sel = st.session_state.selected_match
    season = sel["season"]
    gw = sel["gameweek"]
    
    df_season = df[df["Season"] == season].copy()
    if df_season.empty:
        st.error(f"No data for season {season}.")
        return

    # Load the match data
    home_row = df_season.loc[sel["home_index"]]
    away_row = df_season.loc[sel["away_index"]]

    home_disp = to_display_series(home_row)
    away_disp = to_display_series(away_row)

    st.markdown("---")

    # Predicted vs actual form labels
    home_pred_label = int(home_row["PredLabel"])
    away_pred_label = int(away_row["PredLabel"])
    home_pred_proba = float(home_row["PredProba"])
    away_pred_proba = float(away_row["PredProba"])

    home_actual = int(home_row["FormLabel"])
    away_actual = int(away_row["FormLabel"])

    st.markdown(
        f"### {sel['Home']} vs {sel['Away']} "
        f"({sel['MatchDate'].date()}, GW {gw})"
    )

    home_goals = int(home_disp["GoalsFor"])
    away_goals = int(home_disp["GoalsAgainst"])

    st.markdown("#### Final Score")
    score_home_col, score_mid_col, score_away_col = st.columns([3, 2, 3])

    home_logo = get_logo_path(sel["Home"])
    away_logo = get_logo_path(sel["Away"])

    def render_team_block(team_name, logo_path):
        if logo_path:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<img src='data:image/png;base64,{to_base64(logo_path)}' "
                f"style='width:80px;height:auto;'/></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='text-align:center;font-size:2.5rem;font-weight:700;'>"
            f"{team_name}</div>",
            unsafe_allow_html=True,
        )

    with score_home_col:
        render_team_block(sel["Home"], home_logo)
    with score_mid_col:
        st.markdown(
            f"<div style='text-align:center;padding:0.25rem 0;"
            f"font-size:2.5rem;font-weight:700;'>"
            f"{home_goals} - {away_goals}</div>",
            unsafe_allow_html=True,
        )
    with score_away_col:
        render_team_block(sel["Away"], away_logo)

    st.markdown("---")
    st.markdown("#### Team Form Check (Predicted vs Actual)")

    team_form_rows = [
        {
            "team": sel["Home"],
            "pred_label": home_pred_label,
            "pred_proba": home_pred_proba,
            "actual": home_actual,
            "last5": last5_record_string(df, season, sel["Home"], sel["MatchDate"]),
            "h2h": f"{home_disp['h2h_win_rate']*100:.2f}%",
        },
        {
            "team": sel["Away"],
            "pred_label": away_pred_label,
            "pred_proba": away_pred_proba,
            "actual": away_actual,
            "last5": last5_record_string(df, season, sel["Away"], sel["MatchDate"]),
            "h2h": f"{away_disp['h2h_win_rate']*100:.2f}%",
        },
    ]

    col_team_home, col_team_away = st.columns(2)
    for col, row in zip([col_team_home, col_team_away], team_form_rows):
        predicted_status = "On-form" if row["pred_label"] == 1 else "Off-form"
        actual_status = "On-form" if row["actual"] == 1 else "Off-form"
        correct = row["pred_label"] == row["actual"]
        with col:
            st.markdown(f"**{row['team']}**")
            st.metric(
                "Predicted Form",
                predicted_status,
                f"{row['pred_proba']*100:.2f}% confidence",
            )
            st.metric("Actual Form", actual_status)
            if correct:
                st.success("Prediction correctly matched the actual form.")
            else:
                st.error("Prediction did not match the actual form.")
            st.write(f"Last 5 results: {row['last5']}")
            st.write(f"Head-to-Head win rate vs opponent: {row['h2h']}")

    match_accuracy = sum(1 for r in team_form_rows if r["pred_label"] == r["actual"])
    match_accuracy_pct = match_accuracy / len(team_form_rows) * 100
    st.metric("Form prediction accuracy for this match", f"{match_accuracy_pct:.2f}%", f"{match_accuracy}/2 correct")

    form_comparison_df = pd.DataFrame([
        {
            "Team": row["team"],
            "Predicted": "On-form" if row["pred_label"] == 1 else "Off-form",
            "Actual": "On-form" if row["actual"] == 1 else "Off-form",
            "Confidence": f"{row['pred_proba']*100:.2f}%",
            "Correct?": "Yes" if row["pred_label"] == row["actual"] else "No",
        }
        for row in team_form_rows
    ])
    st.dataframe(form_comparison_df, use_container_width=True, hide_index=True)

    st.markdown("#### Raw Match Stats")

    team_stats = {
        "Goals": home_disp["GoalsFor"],
        "Goals Conceded": home_disp["GoalsAgainst"],
        "Shots": home_disp["ShotsFor"],
        "Shots on Target": home_disp["ShotsOnTargetFor"],
        "Corners": home_disp["CornersFor"],
        "Fouls": home_disp["FoulsFor"],
        "Yellow Cards": home_disp["YellowCards"],
        "Red Cards": home_disp["RedCards"],
    }

    opp_stats = {
        "Goals": away_disp["GoalsFor"],
        "Goals Conceded": away_disp["GoalsAgainst"],
        "Shots": away_disp["ShotsFor"],
        "Shots on Target": away_disp["ShotsOnTargetFor"],
        "Corners": away_disp["CornersFor"],
        "Fouls": away_disp["FoulsFor"],
        "Yellow Cards": away_disp["YellowCards"],
        "Red Cards": away_disp["RedCards"],
    }

    stats_rows = []
    for metric in team_stats.keys():
        stats_rows.append({
            sel["Home"]: team_stats[metric],
            "Metric": metric,
            sel["Away"]: opp_stats[metric],
        })
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(
        stats_df[[sel["Home"], "Metric", sel["Away"]]],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Use this review to validate the form prediction model by comparing predicted versus actual form labels and inspecting raw match statistics for deeper insights."
    )

    # Back to Dashboard button at the bottom
    st.markdown("---")
    if st.button("⬅️ Back to Match Watchability", type="primary", use_container_width=True):
        # Deactivate Match Result Analysis mode
        st.session_state.match_result_mode = False
        # Switch back to Match Watchability
        st.session_state.current_page = "2. Match Watchability"
        st.rerun()


# ---------------------------------------
# PAGE 4 – Club Stats
# ---------------------------------------
def page_club_stats(df: pd.DataFrame):
    st.title("📈 Club Stats")

    if "selected_season" not in st.session_state or st.session_state.selected_season is None:
        st.warning("Please go to **Select Season & Club** first to choose a season.")
        return

    season = st.session_state.selected_season
    fav_clubs = st.session_state.get("fav_clubs", [])
    view_mode = st.session_state.get("view_mode", "All fixtures")
    selected_gameweek = st.session_state.get("selected_gameweek", None)

    df_season = df[df["Season"] == season].copy()
    if df_season.empty:
        st.error(f"No data for season {season}.")
        return

    # Filter by gameweek if selected in Dashboard (show completed matches only: week-1)
    display_gameweek = None
    if selected_gameweek is not None:
        # Show stats for completed matches only (up to week before selected)
        display_gameweek = max(0, selected_gameweek - 1)
        if display_gameweek > 0:
            df_season = df_season[df_season["season_week"] <= display_gameweek].copy()
            st.info(f"Showing stats up to Week {display_gameweek} (completed matches before Gameweek {selected_gameweek})")
        else:
            # No completed matches yet
            df_season = df_season[df_season["season_week"] < 1].copy()  # Empty dataset
            st.info(f"Gameweek {selected_gameweek} selected in Dashboard. No matches completed yet.")

    # Get teams based on Dashboard preferences
    teams = get_teams_for_season(df, season)

    # Filter teams based on view mode from Dashboard
    if view_mode == "Favourite clubs only" and fav_clubs:
        available_teams = [t for t in teams if t in fav_clubs]
        if not available_teams:
            st.warning("No favourite clubs selected. Showing all teams.")
            available_teams = teams
        else:
            st.info(f"Showing stats for favourite clubs only (as selected in Dashboard)")
    else:
        available_teams = teams

    club = st.selectbox("Select club:", available_teams, index=0)
    club_df = df_season[df_season["Team"] == club].copy()
    if club_df.empty:
        st.info("No data available for this club in the selected season/gameweek range.")
        return

    # Display gameweek info based on completed matches
    if display_gameweek is not None:
        if display_gameweek > 0:
            gameweek_info = f" (up to Week {display_gameweek})"
        else:
            gameweek_info = " (no matches played yet)"
    else:
        gameweek_info = ""
    st.markdown(f"### {club} – Season Summary ({season}{gameweek_info})")

    # W/D/L and goals
    result_codes = club_df["Result"].apply(result_to_code)
    total_matches = len(club_df)
    wins = (result_codes == "W").sum()
    draws = (result_codes == "D").sum()
    losses = (result_codes == "L").sum()
    goals_for = club_df["GoalsFor"].sum()
    goals_against = club_df["GoalsAgainst"].sum()
    win_rate = wins / total_matches if total_matches > 0 else 0.0

    avg_mpi = club_df["MPI"].mean()
    on_form = (club_df["FormLabel"] == 1).sum()
    off_form = (club_df["FormLabel"] == 0).sum()

    # Model accuracy for this club
    correct = (club_df["PredLabel"] == club_df["FormLabel"]).sum()
    acc = correct / total_matches if total_matches > 0 else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Played", total_matches)
    c2.metric("W / D / L", f"{wins} / {draws} / {losses}")
    c3.metric("Season Win Rate", f"{win_rate*100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Goals For / Against", f"{goals_for} / {goals_against}")
    c5.metric("Average MPI", f"{avg_mpi:.2f}")
    c6.metric("On-Form vs Off-Form (Actual)", f"{on_form} / {off_form}")

    if acc is not None:
        st.metric("Model Prediction Accuracy for This Club", f"{acc*100:.2f}%")

    st.markdown("#### Result Breakdown vs Predicted Form")
    pred_label_map = {1: "Predicted On-form", 0: "Predicted Off-form"}
    pred_vs_results = (
        club_df.assign(
            ResultCode=club_df["Result"].apply(result_to_code),
            PredForm=club_df["PredLabel"].map(pred_label_map).fillna("Predicted Off-form")
        )
        .groupby("PredForm")["ResultCode"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in ["W", "D", "L"]:
        if col not in pred_vs_results.columns:
            pred_vs_results[col] = 0
    pred_vs_results = pred_vs_results[["W", "D", "L"]]
    pred_vs_results = pred_vs_results.reindex(
        ["Predicted On-form", "Predicted Off-form"]
    ).fillna(0)
    pred_vs_results[["W", "D", "L"]] = pred_vs_results[["W", "D", "L"]].astype(int)
    pred_vs_results["Matches"] = pred_vs_results[["W", "D", "L"]].sum(axis=1)
    pred_vs_results["Win %"] = pred_vs_results.apply(
        lambda row: (row["W"] / row["Matches"] * 100) if row["Matches"] else 0.0,
        axis=1,
    ).round(2)
    st.dataframe(pred_vs_results, use_container_width=True)
    st.caption("Shows actual Win/Draw/Loss outcomes categorized by whether the model predicted this club as on-form or off-form.")

    st.markdown("#### Aggregated Match Statistics (This Season)")
    agg_stats = {
        "Total Shots": club_df["ShotsFor"].sum(),
        "Total Shots on Target": club_df["ShotsOnTargetFor"].sum(),
        "Total Corners": club_df["CornersFor"].sum(),
        "Total Fouls": club_df["FoulsFor"].sum(),
        "Total Yellow Cards": club_df["YellowCards"].sum(),
        "Total Red Cards": club_df["RedCards"].sum(),
    }
    st.table(pd.DataFrame(agg_stats, index=["Value"]).T)

    st.markdown("#### Match-by-Match Details")
    show_cols = [
        "MatchDate", "season_week", "Opponent",
        "GoalsFor", "GoalsAgainst", "Result",
        "PredProba", "PredLabel", "FormLabel", "MPI",
        "ShotsFor", "ShotsOnTargetFor",
        "CornersFor", "FoulsFor",
        "YellowCards", "RedCards",
    ]
    available_cols = [c for c in show_cols if c in club_df.columns]
    tmp = club_df[available_cols].copy()
    tmp = tmp.sort_values("MatchDate")
    st.dataframe(tmp, use_container_width=True)

    st.caption(
        "This view summarizes the entertainment value of this club's matches "
        "(based on goals, shots, and form) and evaluates the watchability model's reliability for this club. "
        "Statistics are filtered based on your Dashboard preferences (season, gameweek, and view mode)."
    )


# ---------------------------------------
# PAGE 5 – League Standings
# ---------------------------------------
def page_league_table(df: pd.DataFrame):
    st.title("🏆 League Standings")

    if "selected_season" not in st.session_state or st.session_state.selected_season is None:
        st.warning("Please go to **Select Season & Club** first to choose a season.")
        return

    # Use selected season from session state
    season = st.session_state.selected_season
    selected_gameweek = st.session_state.get("selected_gameweek", None)

    df_season = df[df["Season"] == season].copy()
    if df_season.empty:
        st.error(f"No data for season {season}.")
        return

    # Calculate the gameweek to display (completed matches only)
    # If gameweek 1 is selected, show week 0 (no matches played yet)
    if selected_gameweek is not None:
        display_gw = max(0, selected_gameweek - 1)
    else:
        # If no gameweek selected, show full season
        max_week = int(df_season["season_week"].max())
        display_gw = max_week

    st.info(f"Showing league table for season: **{season}**")
    if selected_gameweek is not None:
        st.info(f"Gameweek **{selected_gameweek}** selected in Dashboard. Showing standings after **Week {display_gw}** (completed matches only).")

    if display_gw == 0:
        st.markdown(f"### League Standings - {season} (No Matches Played Yet)")
        st.info("No matches have been completed yet. The league table will be available after Gameweek 1 matches are played.")
    else:
        st.markdown(f"### League Standings - {season} (After Week {display_gw})")
        table = build_league_table_up_to_week(df, season, display_gw)
        if table.empty:
            st.info("No league data is available.")
        else:
            st.dataframe(table, use_container_width=True)
            trend_df = build_position_trends(df, season, display_gw)
            if trend_df.empty:
                st.info("Insufficient data available to plot league position changes at this time.")
            else:
                st.markdown("#### League Position Changes Over Time")
                teams_available = sorted(trend_df["Team"].unique())
                fav_defaults = [
                    club for club in st.session_state.get("fav_clubs", [])
                    if club in teams_available
                ]
                default_selection = fav_defaults or teams_available[: min(6, len(teams_available))]
                selected = st.multiselect(
                    "Select clubs to plot",
                    teams_available,
                    default=default_selection,
                    key=f"trend_clubs_{season}",
                )
                if not selected:
                    st.info("Select at least one club to see the time-series chart.")
                else:
                    chart_df = trend_df[trend_df["Team"].isin(selected)]
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "Week:Q",
                                title="Gameweek",
                                scale=alt.Scale(domain=[1, display_gw], nice=False, zero=False),
                                axis=alt.Axis(format="d", tickMinStep=1),
                            ),
                            y=alt.Y(
                                "Pos:Q",
                                title="League Position",
                                scale=alt.Scale(domain=[20, 1], nice=False),
                                axis=alt.Axis(values=list(range(1, 21))),
                            ),
                            color=alt.Color("Team:N", title="Club"),
                            tooltip=[
                                alt.Tooltip("Team:N", title="Club"),
                                alt.Tooltip("Week:Q", title="Week"),
                                alt.Tooltip("Pos:Q", title="Position"),
                            ],
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "This table displays standings based on completed matches only. "
        "The season is selected from **Select Season & Club**, and the gameweek follows your **Dashboard** selection."
    )


# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    st.set_page_config(
        page_title="EPL Match Watchability",
        layout="wide",
    )

    df = load_data_with_predictions(DATA_PATH, MODEL_PATH)

    # Initialize current_page in session state if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "0. User Guide"

    # Initialize match_result_mode flag
    if "match_result_mode" not in st.session_state:
        st.session_state.match_result_mode = False

    # Check if we're in Match Result Analysis mode
    in_match_result_mode = st.session_state.match_result_mode

    # Only show sidebar navigation when NOT in Match Result Analysis mode
    if not in_match_result_mode:
        st.sidebar.title("Navigation")

        # Use session state to control the selected page
        page_options = [
            "0. User Guide",
            "1. Select Season & Club",
            "2. Match Watchability",
            "4. Club Stats",
            "5. League Standings",
        ]

        # Get the index of current page
        current_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0

        page = st.sidebar.radio(
            "Go to",
            page_options,
            index=current_index,
        )

        # Update current_page if user manually selects from sidebar
        if page != st.session_state.current_page:
            st.session_state.current_page = page

        # Add helpful info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.info("💡 New to the dashboard? Check out the **User Guide** for detailed instructions!")

        if page.startswith("0."):
            page_user_guide()
        elif page.startswith("1."):
            page_setup(df)
        elif page.startswith("2."):
            page_watchability(df)
        elif page.startswith("4."):
            page_club_stats(df)
        elif page.startswith("5."):
            page_league_table(df)
    else:
        # In Match Result Analysis mode - only show this page
        page_match_result(df)


if __name__ == "__main__":
    main()
