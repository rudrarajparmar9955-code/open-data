# analysis_and_export.py

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def calculate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates aggregate statistics for each striker."""

    # 1. Define outcome categories
    df['is_goal'] = df['penalty_outcome'].isin(['Goal'])
    df['is_miss'] = df['penalty_outcome'].isin(['Off T', 'Post'])
    df['is_saved'] = df['penalty_outcome'].isin(['Saved'])

    # 2. Group by Striker
    player_stats = df.groupby('striker').agg(
        penalties_taken=('striker', 'size'),
        penalties_scored=('is_goal', 'sum'),
        penalties_missed=('is_miss', 'sum'),
        penalties_saved_by_keeper=('is_saved', 'sum')
    ).reset_index()

    # 3. Calculate Rates
    player_stats['conversion_rate'] = (
            player_stats['penalties_scored'] / player_stats['penalties_taken']
    ).fillna(0)

    player_stats['penalty_goals_percent'] = player_stats['conversion_rate']

    footedness_lookup = df[['striker', 'footedness']].drop_duplicates(subset=['striker'])
    player_stats = player_stats.merge(footedness_lookup, on='striker', how='left')

    final_cols = ['striker', 'footedness', 'penalties_taken', 'penalties_scored',
                  'penalties_missed', 'conversion_rate', 'penalty_goals_percent']

    return player_stats[final_cols]


def calculate_keeper_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates aggregate statistics for each goalkeeper."""

    keeper_actions = df.copy()

    # Identify penalties taken AGAINST this keeper
    keeper_stats = keeper_actions.groupby('goalkeeper_name').agg(
        penalties_taken_against=('match_id', 'size'),
        penalties_saved=('is_saved', 'sum'),
        goals_conceded=('is_goal', 'sum')
    ).reset_index()

    # Calculate Save Rate
    keeper_stats['goalkeeper_save_rate'] = (
            keeper_stats['penalties_saved'] / keeper_stats['penalties_taken_against']
    ).fillna(0)

    return keeper_stats.rename(columns={'goalkeeper_name': 'goalkeeper'})


def export_data(all_penalties: List[Dict[str, Any]], output_path: Path):
    """
    Converts raw penalty events to a DataFrame, sorts it, and writes three output CSVs.
    """

    if not all_penalties:
        print("No penalty data extracted. Exiting.")
        return

    df_raw = pd.DataFrame(all_penalties)

    # --- 1. Raw Event Log (Sorted) ---

    # Calculate total seconds into the match for precise ordering
    df_raw['match_time_seconds'] = df_raw['minute'] * 60 + df_raw['second']

    # Convert match_date to datetime for proper chronological sorting
    df_raw['match_date'] = pd.to_datetime(df_raw['match_date'])

    # Sort by League (alphabetical), Date (oldest to latest), and then Match Time
    df_raw_sorted = df_raw.sort_values(
        by=['league_name', 'match_date', 'match_time_seconds'],
        ascending=[True, True, True]
    ).drop(columns=['match_time_seconds', 'is_goal', 'is_miss', 'is_saved'], errors='ignore')

    df_raw_sorted.to_csv(output_path / "penalty_event_log.csv", index=False)
    print(f"Wrote {len(df_raw_sorted)} raw penalty events to penalty_event_log.csv (Sorted by League and Time)")

    # --- 2. Player Stats ---
    df_players = calculate_player_stats(df_raw)
    df_players.to_csv(output_path / "player_penalty_summary.csv", index=False)
    print(f"Wrote {len(df_players)} player summaries to player_penalty_summary.csv")

    # --- 3. Goalkeeper Stats ---
    df_keepers = calculate_keeper_stats(df_raw)
    df_keepers.to_csv(output_path / "keeper_penalty_summary.csv", index=False)
    print(f"Wrote {len(df_keepers)} goalkeeper summaries to keeper_penalty_summary.csv")
