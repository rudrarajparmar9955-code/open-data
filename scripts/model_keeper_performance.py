# model_keeper_performance.py
# GOAL: Create CORE MODEL 3: Goalkeeper Performance Model.
# This script loads the two trained models (Goal Conversion and xG) to create
# final features and trains a model to predict the penalty outcome using all available information.
# This serves as the most complete penalty prediction model.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Any, Optional, List

# Set the base directory (TEST/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"
XG_MODEL_PATH = OUTPUT_DIR / "expected_goals_model.pkl"
CONVERSION_MODEL_PATH = OUTPUT_DIR / "goal_conversion_final_model.pkl"


# --- Placeholder for ELO Calculation (must be consistent with model 1) ---

def calculate_elo_rating(df: pd.DataFrame, default_elo=1500, k_factor=32) -> pd.DataFrame:
    """Calculates dynamic Elo ratings for strikers and keepers."""
    KEEPER_COLUMN = 'goalkeeper_name'
    # IMPORTANT: Sort by date/time to ensure ELO is calculated chronologically
    df = df.sort_values(by=['match_date', 'minute', 'second']).reset_index(drop=True)
    striker_elos = {}
    keeper_elos = {}
    striker_elo_results = []
    keeper_elo_results = []
    df['outcome'] = (df['penalty_outcome'] == 'Goal').astype(float)

    for index, row in df.iterrows():
        striker = row['striker']
        keeper = row[KEEPER_COLUMN]
        outcome = row['outcome']

        striker_elo = striker_elos.get(striker, default_elo)
        keeper_elo = keeper_elos.get(keeper, default_elo)

        striker_elo_results.append(striker_elo)
        keeper_elo_results.append(keeper_elo)

        # Expected outcome for striker
        E_striker = 1 / (1 + 10 ** ((keeper_elo - striker_elo) / 400))

        # Update ELOs
        striker_new_elo = striker_elo + k_factor * (outcome - E_striker)
        keeper_new_elo = keeper_elo + k_factor * ((1 - outcome) - (1 - E_striker))

        striker_elos[striker] = striker_new_elo
        keeper_elos[keeper] = keeper_new_elo

    elo_df = pd.DataFrame({
        'striker_elo_prev': striker_elo_results,
        'keeper_elo_prev': keeper_elo_results
    })
    return df.join(elo_df)


# --- Feature Preparation and Prediction Function ---

def prepare_and_predict_features(df: pd.DataFrame, xg_model: Pipeline, conv_model: Pipeline) -> pd.DataFrame:
    """
    Calculates ELO, generates predictions (xG and ELO Prob), and filters data.
    """

    # 1. Calculate ELO features
    df_engineered = calculate_elo_rating(df)

    # 2. Filter for known ELO and valid xG data (shot_angle)
    # The keeper performance analysis relies on having both xG and ELO.
    df_model = df_engineered[
        (df_engineered['striker_elo_prev'] != 1500) &
        (df_engineered['shot_angle'].notna())
        ].copy()

    df_model['target'] = (df_model['penalty_outcome'] == 'Goal').astype(int)

    # === XG Model Features (Geometric Focus) ===
    xg_features = [
        'shot_angle', 'shot_side', 'shot_height',
        'footedness', 'match_scoreline_diff', 'home_or_away'
    ]
    X_xg = df_model[xg_features]

    # Predict xG (Probability of Goal based purely on shot geometry)
    df_model['xg_probability'] = xg_model.predict_proba(X_xg)[:, 1]

    # === Conversion Model Features (ELO/Context Focus) ===
    conv_features = [
        'footedness', 'minute', 'match_scoreline_diff',
        'home_or_away'
    ]
    # NOTE: 'elo_advantage' is calculated inside the conversion model's preprocessor
    # as a combination of 'striker_elo_prev' and 'keeper_elo_prev', but since the
    # original model used the engineered 'elo_advantage' feature which is not available
    # in the Conv Model's fitting data, we need to re-engineer it here
    # for consistent structure (even though the L1 model zeroed out most features).
    df_model['elo_advantage'] = df_model['striker_elo_prev'] - df_model['keeper_elo_prev']
    conv_features.append('elo_advantage')

    X_conv = df_model[conv_features]

    # Predict ELO Prob (Probability of Goal based purely on player skill/context)
    # NOTE: We can skip predicting with the old conv_model as its score was poor
    # and its best features (ELO/Scoreline) are now directly used in the final combined model.
    # Instead of predicting, we will just use the ELO advantage as a feature.

    return df_model


def calculate_keeper_gpaa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Goals Prevented Above Average (GPAA) for each goalkeeper.
    GPAA = Total Expected Goals Conceded - Actual Goals Conceded.
    A positive GPAA means the keeper saved more than expected.
    """

    # Aggregate stats by keeper
    keeper_stats = df.groupby('goalkeeper_name').agg(
        penalties_faced=('target', 'size'),
        actual_goals_conceded=('target', 'sum'),
        total_expected_goals=('xg_probability', 'sum')  # Sum of xG values is Total xG Faced
    ).reset_index()

    # Calculate GPAA
    keeper_stats['gpaa'] = keeper_stats['total_expected_goals'] - keeper_stats['actual_goals_conceded']

    # Sort by GPAA (Highest positive is best)
    keeper_stats = keeper_stats.sort_values(by='gpaa', ascending=False)

    final_cols = ['goalkeeper_name', 'penalties_faced', 'total_expected_goals',
                  'actual_goals_conceded', 'gpaa']

    return keeper_stats[final_cols].rename(columns={'goalkeeper_name': 'goalkeeper'})


def train_and_evaluate_combined_model(df: pd.DataFrame):
    """
    Trains the final, comprehensive penalty model using both xG and ELO features.
    """

    # --- Load Pre-Trained Models ---
    if not XG_MODEL_PATH.exists() or not CONVERSION_MODEL_PATH.exists():
        print(f"Error: Required model files not found. Run previous scripts first.")
        print(f"Missing: {XG_MODEL_PATH.name} or {CONVERSION_MODEL_PATH.name}")
        return

    xg_model = joblib.load(XG_MODEL_PATH)
    conv_model = joblib.load(CONVERSION_MODEL_PATH)

    # --- Prepare Data with Predictions ---
    df_combined = prepare_and_predict_features(df, xg_model, conv_model)

    # --- Feature Set for CORE MODEL 3 (Comprehensive) ---
    # This model uses the derived xG and ELO Advantage directly as primary features
    features = [
        'xg_probability',  # Prediction from Model 2 (Geometric likelihood)
        'elo_advantage',  # Skill differential (Contextual likelihood)
        'shot_side',  # Where the shot went
        'shot_height',  # How high the shot went
    ]

    X = df_combined[features]
    y_encoded = df_combined['target']

    # Class 0: 'Not Goal', Class 1: 'Goal'
    target_names = ['Not Goal', 'Goal']

    print(f"\nTotal model samples (filtered for Combined Model): {len(X)} samples")
    print(f"Target Labels (0, 1): {target_names}")

    # --- Create Preprocessing Pipeline ---
    categorical_features = ['shot_side', 'shot_height']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with L2 Logistic Regression (Robust) ---
    classifier = LogisticRegression(
        penalty='l2',  # L2 regularization for robustness
        class_weight='balanced',  # CRITICAL for imbalance
        random_state=42,
        max_iter=1000
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # --- Training and Cross-Validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n--- Training CORE MODEL 3 (Goalkeeper Performance/Combined Prediction) ---")

    # Train the final model on all data
    model_pipeline.fit(X, y_encoded)
    best_model = model_pipeline

    # Calculate Cross-Validation Score (ROC AUC)
    cv_scores = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_score = np.mean(cv_scores)

    # --- Evaluation ---
    y_prob = best_model.predict_proba(X)[:, 1]  # Get probabilities for the 'Goal' class

    # 1. ROC AUC on Full Dataset
    roc_auc_full = roc_auc_score(y_encoded, y_prob)
    print(f"\nModel ROC AUC on Full Dataset: {roc_auc_full:.4f}")

    # 2. K-Fold Cross-Validated Score (Primary CV metric)
    print(f"Cross-Validated ROC AUC Score (5-Fold Mean): {cv_score:.4f} (Targeting > 0.65 for stability)")

    # ----------------------------------------------------
    # Calculate Goalkeeper Performance Metrics
    # ----------------------------------------------------
    keeper_gpaa_df = calculate_keeper_gpaa(df_combined)

    print("\n--- Goalkeeper Performance (Top 5 GPAA) ---")
    print("GPAA (Goals Prevented Above Average) is positive when the keeper saved more than expected.")
    print(keeper_gpaa_df.head().to_string(index=False))

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------
    MODEL_PATH = OUTPUT_DIR / "keeper_performance_model.pkl"
    joblib.dump(best_model, MODEL_PATH)
    keeper_gpaa_df.to_csv(OUTPUT_DIR / "goalkeeper_gpaa_summary.csv", index=False)

    print("\n--- Model Persistence ---")
    print(f"CORE MODEL 3 Pipeline saved to: {MODEL_PATH}")
    print(f"Goalkeeper GPAA Summary saved to goalkeeper_gpaa_summary.csv")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print("CORE MODEL 3: Logistic Regression BINARY CLASSIFICATION (Combined Prediction & Keeper Analysis)")


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run 'scripts/main.py' first to generate the 'penalty_event_log.csv'.")
    else:
        # Load the data
        raw_df = pd.read_csv(INPUT_FILE)

        # Train, Predict, and Evaluate
        train_and_evaluate_combined_model(raw_df)
