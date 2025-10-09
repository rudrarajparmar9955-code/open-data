# penalty_scouting_report_generator.py
# GOAL: Final utility script that loads all trained models and generates actionable
# scouting intelligence by combining Shot Tendency, xG, ELO, and Goalkeeper GPAA.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any, List, Optional
import sys
import os

# Set the base directory (TEST/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"

# ------------------------------------------------------------------
# --- CONFIGURATION: EDIT THIS STRIKER NAME BEFORE RUNNING ---
# Enter the full name exactly as it appears in the 'striker' column of your data.
STRIKER_TO_ANALYZE = "Lionel Andrés Messi Cuccittini"
# ------------------------------------------------------------------

# --- MODEL PATHS (Confirmed Best-Performing Models) ---
# CORE MODEL 2: Used for the striker's average shot quality (xG)
XG_MODEL_PATH = OUTPUT_DIR / "expected_goals_model.pkl"

# Best performing model for L vs R prediction (ignoring the poor performing Center class)
SHOT_TENDENCY_MODEL_PATH = OUTPUT_DIR / "shot_tendency_lean_binary_model.pkl"

# Output from CORE MODEL 3: Used for keeper counter-scouting
GPAA_SUMMARY_FILE = OUTPUT_DIR / "goalkeeper_gpaa_summary.csv"


# --- ELO Calculation Function (Must match the logic used in Model 1) ---
def calculate_elo_rating(df: pd.DataFrame, default_elo=1500, k_factor=32) -> pd.DataFrame:
    """Calculates dynamic Elo ratings for strikers and keepers."""
    KEEPER_COLUMN = 'goalkeeper_name'
    # Ensure the data is sorted chronologically for correct ELO update
    df = df.sort_values(by=['match_date', 'minute', 'second']).reset_index(drop=True)
    striker_elos = {}
    keeper_elos = {}
    striker_elo_results = []

    # 'outcome' = 1 for Goal, 0 for Not Goal
    df['outcome'] = (df['penalty_outcome'] == 'Goal').astype(float)

    for index, row in df.iterrows():
        striker = row['striker']
        keeper = row[KEEPER_COLUMN]

        # Load or initialize ELO
        striker_elo = striker_elos.get(striker, default_elo)

        # Store the current ELO (before the penalty is taken)
        striker_elo_results.append(striker_elo)

        keeper_elo = keeper_elos.get(keeper, default_elo)

        E_striker = 1 / (1 + 10 ** ((keeper_elo - striker_elo) / 400))

        # Update ELOs
        striker_new_elo = striker_elo + k_factor * (row['outcome'] - E_striker)
        keeper_new_elo = keeper_elo + k_factor * ((1 - row['outcome']) - (1 - E_striker))

        striker_elos[striker] = striker_new_elo
        keeper_elos[keeper] = keeper_new_elo

    elo_df = pd.DataFrame({
        'striker_elo_prev': striker_elo_results,
    })
    return df.join(elo_df[['striker_elo_prev']])


# --- Main Prediction and Report Generation Function ---

def generate_scouting_report(striker_name: str):
    """
    Generates a full scouting report for a specified striker.
    """

    # 1. Load Data and Models
    try:
        raw_df = pd.read_csv(INPUT_FILE)
        xg_model = joblib.load(XG_MODEL_PATH)
        shot_tendency_model = joblib.load(SHOT_TENDENCY_MODEL_PATH)
        gpaa_df = pd.read_csv(GPAA_SUMMARY_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure all model scripts were run. Missing: {e.filename}")
        return
    except Exception as e:
        print(f"Error loading models or data: {e}")
        return

    # 2. Prepare Data (Calculate latest ELO)
    df_engineered = calculate_elo_rating(raw_df)

    # --- Calculate xG using the loaded xG model ---
    # The xG model requires all features used during training.
    xg_features = ['shot_angle', 'shot_height', 'match_scoreline_diff', 'footedness', 'home_or_away', 'shot_side']

    # Filter for penalties that have the necessary features for prediction
    df_predict = df_engineered[df_engineered['shot_angle'].notna() & df_engineered['shot_height'].notna()].copy()

    # Initialize the xg_probability column to NaN for all rows
    df_engineered['xg_probability'] = np.nan

    if not df_predict.empty and all(col in df_predict.columns for col in xg_features):
        try:
            # Select features and fill NaNs if any remain (usually for match_scoreline_diff)
            X_xg = df_predict[xg_features].fillna(0)

            # Predict the probability of success (xG)
            df_predict['xg_probability'] = xg_model.predict_proba(X_xg)[:, 1]

            # Update the xg_probability column in the main engineered dataframe
            # Use 'update' to fill only the non-NaN values from df_predict into df_engineered
            df_engineered.update(df_predict[['xg_probability']])

        except Exception as e:
            # Print the error for debugging purposes but continue the script
            print(f"Warning: Could not predict xG with model. Skipping xG calculation. Error: {e}")
    else:
        print("Warning: Missing required features or empty data for xG prediction. Skipping xG calculation.")
    # ---------------------------------------------------

    # Filter for the specific striker's historical penalties
    striker_data = df_engineered[df_engineered['striker'] == striker_name].copy()

    if striker_data.empty:
        print(f"\nNo penalty data found for striker: '{striker_name}'")
        print("Please check the spelling or ensure the player has taken penalties in the dataset.")

        # Display list of available strikers for easy copying
        print("\nAvailable strikers (Top 10 by volume):")
        try:
            top_strikers = raw_df['striker'].value_counts().head(10).index.tolist()
            for s in top_strikers:
                print(f"- {s}")
        except:
            print("- Data loading failed.")
        return

    print("\n" + "=" * 80)
    print(f"| PENALTY SCOUTING REPORT: {striker_name.upper():<55} |")
    print("=" * 80)

    # --- Section 1: Striker Skill and Shot Quality (xG & ELO) ---

    latest_elo = striker_data['striker_elo_prev'].iloc[-1].round(0)

    # Filter for shots that have xG features and calculate average xG
    avg_xg_series = striker_data[striker_data['shot_angle'].notna()]['xg_probability'].mean()
    if pd.isna(avg_xg_series):
        avg_xg = "0.000 (Data Insufficient)"
    else:
        avg_xg = avg_xg_series.round(3)

    print("\n## 1. Striker Profile: Skill and Shot Quality")
    print("-" * 40)
    print(f"• Latest Striker ELO Rating (Skill Score): {latest_elo}")
    print(f"• Average Expected Goals (xG) per shot: {avg_xg} (Measures shot placement quality)")

    # --- Section 2: Shot Direction Tendency (Historical & Actionable Prediction) ---

    print("\n## 2. Shot Direction Tendency & Actionable Lean")
    print("-" * 70)

    # A. Calculate historical tendency (L, C, R)
    historical_tendency = striker_data['shot_side'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

    print("A. Historical Shooting Distribution (All attempts):")
    print(historical_tendency.to_string())

    # B. Actionable Binary Prediction (L vs R only)

    # Features required by the binary shot tendency model (MUST match the error list)
    required_features = [
        'keeper_elo_prev', 'minute', 'match_scoreline_diff',
        'home_or_away', 'hist_left_pct', 'cum_total_prev',
        'elo_advantage', 'footedness'  # These two were also implicitly expected
    ]
    target_mapping = ['Left', 'Right']

    # Get the data for the LAST penalty taken, as a forward-looking model
    # would rely on the situation right before the shot.
    last_penalty = striker_data.iloc[[-1]].copy()

    # Calculate necessary features for the last penalty
    # 1. keeper_elo_prev: Requires knowing the keeper's ELO from the ELO calculation.
    #    Since we don't have keeper_elos dictionary available here, we will have to mock
    #    keeper_elo_prev based on striker_elo_prev.
    #    For simplicity and reliability, we will mock the current keeper's ELO as 1500 (average)
    last_penalty['keeper_elo_prev'] = 1500.0

    # 2. elo_advantage: Calculated as striker_elo_prev - keeper_elo_prev
    last_penalty['elo_advantage'] = last_penalty['striker_elo_prev'] - last_penalty['keeper_elo_prev']

    # 3. cum_total_prev: Total penalties taken by the striker up to the last event.
    last_penalty['cum_total_prev'] = len(striker_data) - 1  # Total count minus current event

    # 4. hist_left_pct: Historical % of shots to the Left before this event
    #    We'll calculate the *overall* historical left percentage for the striker
    left_shots_count = striker_data['shot_side'].value_counts().get('Left', 0)
    total_shots = len(striker_data)
    last_penalty['hist_left_pct'] = (left_shots_count / total_shots) if total_shots > 0 else 0.0

    # Ensure all required features are present and handle potential missingness/types
    for col in required_features:
        if col not in last_penalty.columns:
            # Fallback for truly missing columns from the original data (minute, home_or_away etc.)
            last_penalty[col] = 0.0 if col in ['minute', 'match_scoreline_diff'] else last_penalty.get(col, 'Unknown')

    # Select the final feature vector for prediction
    mock_input = last_penalty[required_features].fillna(0).head(1)

    try:
        # Use predict_proba to get probabilities for the two classes (Left, Right)
        predicted_probs = shot_tendency_model.predict_proba(mock_input)[0]
    except Exception as e:
        print(f"Warning: Could not predict Shot Tendency. Error: {e}")
        # Fallback if prediction fails
        predicted_probs = [0.5, 0.5]

    prediction_df = pd.DataFrame({
        'Direction': target_mapping,
        'Probability': predicted_probs
    }).sort_values(by='Probability', ascending=False)

    prediction_df['Probability'] = (prediction_df['Probability'] * 100).round(1).astype(str) + '%'

    most_likely_side = prediction_df['Direction'].iloc[0]
    most_likely_prob = prediction_df['Probability'].iloc[0]

    print("\nB. Actionable Prediction (Left vs. Right Only):")
    print("NOTE: This uses the most stable model to suggest a side preference.")
    print(prediction_df.to_string(index=False))

    # --- Section 3: Goalkeeper Counter-Scouting (GPAA) ---

    top_keepers = gpaa_df.head(5).copy()

    print("\n## 3. Goalkeeper Counter-Scouting (Top Performers)")
    print("-" * 70)
    print("Top 5 Keepers by Goals Prevented Above Average (GPAA):")
    print(top_keepers.rename(columns={'gpaa': 'GPAA (Goals Prevented)',
                                      'total_expected_goals': 'xG Faced',
                                      'actual_goals_conceded': 'Goals Conceded'}).to_string(index=False))

    # --- D. Actionable Summary ---

    print("\n## Final Actionable Summary for the Keeper")
    print("-" * 80)
    print(
        f"**PRIMARY TENDENCY:** When this striker avoids the center, they are **{most_likely_prob}** likely to shoot towards the **{most_likely_side.upper()}** side of the goal.")
    print(f"**RECOMMENDATION:** The keeper should favor the **{most_likely_side.upper()}** side when committing.")
    print(
        f"**SHOT QUALITY (xG):** The striker's average shot quality is high (**{avg_xg}**), meaning missteps will likely result in a goal.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # If the default striker name is not set, we cannot run the report.
    if not STRIKER_TO_ANALYZE or STRIKER_TO_ANALYZE == "":
        print("Error: Please set the STRIKER_TO_ANALYZE variable at the top of the script with a player's full name.")
    else:
        # The report now runs directly using the configured name.
        generate_scouting_report(STRIKER_TO_ANALYZE)
