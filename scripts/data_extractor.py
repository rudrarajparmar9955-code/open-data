# data_extractor.py

import json
import math
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

# --- Constants (StatsBomb Pitch Dimensions) ---
# Standard pitch length: 120 units (x-axis)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER_Y = PITCH_WIDTH / 2.0  # 40.0
TOLERANCE_Y = 3.0  # Tolerance for center classification
LOOKAHEAD = 8  # Events to check immediately after a shot for a keeper action


def load_json_file(file_path: Path) -> Optional[Any]:
    """Loads a JSON file and returns its content."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None


def classify_direction_from_y(y: Optional[float]) -> str:
    """Classifies shot direction based on y-coordinate relative to goal center (Y=40)."""
    if y is None:
        return "Unknown"

    if y > GOAL_CENTER_Y + TOLERANCE_Y:
        return "Left"
    elif y < GOAL_CENTER_Y - TOLERANCE_Y:
        return "Right"
    else:
        return "Center"


def classify_height_from_z(z: Optional[float]) -> str:
    """Classifies shot height based on z-coordinate (height off the ground)."""
    if z is None:
        return "Unknown"
    elif z >= 1.5:
        return "Upper"
    elif z >= 0.5:
        return "Middle"
    else:
        return "Lower"


def calculate_shot_angle(x: float, y: float) -> Optional[float]:
    """
    Calculates the angle of the shot, assuming the goal is between
    (120, 36.6) and (120, 43.4).
    """
    if x is None or y is None:
        return None

    Y1 = 36.6
    Y2 = 43.4

    theta1 = math.atan2(abs(Y1 - y), PITCH_LENGTH - x)
    theta2 = math.atan2(abs(Y2 - y), PITCH_LENGTH - x)

    return math.degrees(abs(theta2 - theta1))


def keeper_from_event(ev: Dict[str, Any]) -> Optional[str]:
    """Tries to identify the goalkeeper involved in an event."""
    player_name = ev.get("player", {}).get("name")

    t_name = ev.get("type", {}).get("name", "")
    if any(k in t_name for k in ["Save", "Goalkeeper", "Penalty Saved"]):
        return player_name

    pos_name = ev.get("position", {}).get("name", "")
    if "Goalkeeper" in pos_name:
        return player_name

    if ev.get("goalkeeper"):
        return player_name

    return None


def find_keeper_name(events: List[Dict[str, Any]], idx: int) -> Optional[str]:
    """Searches for the keeper's name immediately after the penalty event using lookahead and related_events."""

    for j in range(1, LOOKAHEAD + 1):
        if idx + j >= len(events):
            break
        cand = events[idx + j]
        k = keeper_from_event(cand)
        if k:
            return k

    shot_ev = events[idx]
    shot_id = shot_ev.get("id")
    for k_ev in events:
        rel = k_ev.get("related_events") or []
        if shot_id in rel:
            k = keeper_from_event(k_ev)
            if k:
                return k
    return None


def extract_penalty_data(match_id: int, base_data_path: Path, competition_data: Dict[int, Dict[str, Any]]) -> List[
    Dict[str, Any]]:
    """
    Extracts all penalty shots and related context for a single match.
    """

    match_events_path = base_data_path / "events" / f"{match_id}.json"
    match_row = competition_data.get(match_id)

    if not match_row:
        return []

    events = load_json_file(match_events_path)
    if not events:
        return []

    comp_name = match_row.get('competition_name', 'Unknown League')

    penalty_rows = []

    for idx, ev in enumerate(events):

        # Check if it is a penalty shot
        if not (ev.get("type", {}).get("name") == "Shot" and ev.get("shot", {}).get("type", {}).get(
                "name") == "Penalty"):
            continue

        # Core Data
        striker = ev.get("player", {}).get("name", "")
        team = ev.get("team", {}).get("name", "")
        minute = ev.get("minute")
        second = ev.get("second")
        shot = ev.get("shot", {})
        outcome = shot.get("outcome", {}).get("name", "")

        # Shot Details
        footedness = shot.get("body_part", {}).get("name", "Unknown")
        end_loc = shot.get("end_location")

        end_x = end_y = end_z = None
        shot_side = "Unknown"
        shot_height = "Unknown"
        shot_angle = None

        if end_loc and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
            end_z = end_loc[2] if len(end_loc) > 2 else None

            shot_side = classify_direction_from_y(end_y)
            shot_height = classify_height_from_z(end_z)
            shot_angle = calculate_shot_angle(end_x, end_y)

        # Goalkeeper Data
        keeper = find_keeper_name(events, idx) or "Unknown Keeper"

        # Match Context
        home_team = match_row.get('home_team', {}).get('home_team_name', 'Unknown Home')
        away_team = match_row.get('away_team', {}).get('away_team_name', 'Unknown Away')

        home_score = match_row.get('home_score', 0)
        away_score = match_row.get('away_score', 0)

        home_or_away = "Home" if team == home_team else "Away"

        scoreline_diff = (home_score - away_score) if team == home_team else (away_score - home_score)

        penalty_rows.append({
            "match_id": match_id,
            "league_name": comp_name,
            "match_date": match_row.get('match_date'),  # NOW INCLUDED for sorting
            "minute": minute,
            "second": second,
            "team": team,
            "striker": striker,
            "footedness": footedness,
            "goalkeeper_name": keeper,
            "penalty_outcome": outcome,
            "shot_side": shot_side,
            "shot_height": shot_height,
            "shot_angle": shot_angle,
            "match_scoreline_diff": scoreline_diff,
            "home_or_away": home_or_away,
        })

    return penalty_rows
#C://Users//Admin//Desktop//TEST//img//logo.png