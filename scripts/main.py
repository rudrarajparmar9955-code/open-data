# main.py

import os
from pathlib import Path
from typing import Dict, Any, List
import data_extractor as de
import analysis_and_export as ae

# Set the base directory (TEST/) by going up one level from the scripts directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"  # Will create this directory


def build_match_data_lookup(comp_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Loads all matches from the matches/ directory, filters for MEN'S competitions,
    and flattens them into a lookup dictionary keyed by match_id.
    """
    match_lookup = {}

    # Iterate through all competition/season IDs in the competitions file
    for comp in comp_data:

        # --- NEW FILTER: Skip non-men's leagues ---
        if comp.get('competition_gender') != 'male':
            # This skips all competitions that are 'female' or null/unknown
            continue
        # ------------------------------------------

        comp_id = comp['competition_id']
        season_id = comp['season_id']

        # Path for the match file (e.g., data/matches/43/106.json)
        match_file_path = DATA_DIR / "matches" / str(comp_id) / f"{season_id}.json"
        matches = de.load_json_file(match_file_path)

        if matches:
            for match in matches:
                match_id = match['match_id']

                # Combine competition data with match data for easy lookup
                match_data = {
                    **match,
                    'competition_name': comp['competition_name'],
                    'season_name': comp['season_name'],
                    'competition_gender': comp['competition_gender']
                }
                match_lookup[match_id] = match_data

    return match_lookup


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. Load All Match Metadata ---
    print("Step 1: Loading all match metadata (Filtering for Men's Leagues)...")
    comp_file = DATA_DIR / "competitions.json"
    comp_data = de.load_json_file(comp_file)

    if not comp_data:
        print("Error: Could not load competition data. Check path or file content.")
        return

    # Create a single dictionary to easily look up match details by ID
    match_data_lookup = build_match_data_lookup(comp_data)
    all_match_ids = list(match_data_lookup.keys())

    if not all_match_ids:
        print("Error: No male matches found. Check 'matches/' directory structure or competition data.")
        return

    print(f"Found {len(all_match_ids)} male matches to process.")

    # --- 2. Iterate and Extract Penalty Data ---
    print("Step 2: Extracting penalty data from all event files...")
    all_penalties = []

    for match_id in all_match_ids:
        # Pass the full match data lookup to the extractor
        penalties = de.extract_penalty_data(match_id, DATA_DIR, match_data_lookup)
        all_penalties.extend(penalties)

    print(f"Extraction Complete. Total penalties found: {len(all_penalties)}")

    # --- 3. Analyze and Export ---
    print("Step 3: Calculating summary statistics and exporting to CSV...")
    # The analysis file now handles the sorting before export
    ae.export_data(all_penalties, OUTPUT_DIR)

    print("\nProcess finished successfully. Check the 'output/' directory for your CSV files.")


if __name__ == "__main__":
    main()
