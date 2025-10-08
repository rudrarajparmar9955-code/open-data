# model_expected_goals.py
# GOAL: Create CORE MODEL 2: Expected Goals (xG) Model.
# This model uses geometric features (angle, side, height) to predict the probability
# of a penalty being scored, which is the formal definition of xG.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Using Gradient Boosting Classifier (GBC) for robust probability estimation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Any

# Set the base directory (TEST/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"


# --- Step 1: Data Preparation ---

def prepare_xg_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw data and filters it for use in the Expected Goals model.
    """
    # Filter out penalties where the shot angle is missing (usually due to bad coordinates)
    df_model = df[df['shot_angle'].notna()].copy()

    # Target Definition: Binary Classification (0=Not Goal, 1=Goal)
    df_model['target'] = (df_model['penalty_outcome'] == 'Goal').astype(int)

    # Filter out events with 'Unknown' keeper/striker (already handled by ELO filter
    # in the previous model, but good practice here)
    df_model = df_model[df_model['goalkeeper_name'] != 'Unknown Keeper']
    df_model = df_model[df_model['striker'] != '']

    return df_model


# --- Step 2: Modeling and Evaluation (Expected Goals - GBC) ---

def train_and_evaluate_xg_model(df: pd.DataFrame):
    """
    Trains a Gradient Boosting Classifier for Expected Goals (xG).
    """

    df_model = prepare_xg_data(df)

    # --- Feature Set for xG ---
    # xG models rely heavily on geometric features. ELO is excluded here to focus
    # purely on the physical factors of the shot.
    features = [
        'shot_angle',  # CRITICAL: Angle of shot to goal mouth
        'shot_side',  # Left/Right/Center placement
        'shot_height',  # Upper/Middle/Lower placement
        'footedness',  # Player's body part used
        'match_scoreline_diff',  # Contextual factor (included for completeness)
        'home_or_away'  # Contextual factor
    ]

    X = df_model[features]
    y_encoded = df_model['target']

    # Class 0: 'Not Goal', Class 1: 'Goal'
    target_names = ['Not Goal', 'Goal']

    print(f"\nTotal model samples (filtered for xG Model): {len(X)} samples")
    print(f"Target Labels (0, 1): {target_names}")

    # --- Create Preprocessing Pipeline ---
    categorical_features = ['shot_side', 'shot_height', 'footedness', 'home_or_away']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with Gradient Boosting Classifier (GBC) ---

    # GBC is generally robust and great for producing calibrated probabilities (xG)
    classifier = GradientBoostingClassifier(
        n_estimators=100,  # Number of trees
        learning_rate=0.1,  # Step size
        max_depth=3,  # Depth of trees (controls complexity)
        random_state=42
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # --- Training and Cross-Validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n--- Training Expected Goals (xG) Model (Gradient Boosting Classifier) ---")

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

    # 2. Brier Score Loss (Measures calibration/accuracy of probabilities)
    brier_loss = brier_score_loss(y_encoded, y_prob)
    print(f"Brier Score Loss (Lower is Better): {brier_loss:.4f}")

    # 3. K-Fold Cross-Validated Score (Primary CV metric)
    print(f"Cross-Validated ROC AUC Score (5-Fold Mean): {cv_score:.4f} (Targeting > 0.65 for stability)")

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------
    MODEL_PATH = OUTPUT_DIR / "expected_goals_model.pkl"
    joblib.dump(best_model, MODEL_PATH)

    print("\n--- Model Persistence ---")
    print(f"Expected Goals Model Pipeline saved to: {MODEL_PATH}")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print("CORE MODEL 2: Gradient Boosting REGRESSION (Expected Goals / xG Probability)")


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run 'scripts/main.py' first to generate the 'penalty_event_log.csv'.")
    else:
        # Load the data
        raw_df = pd.read_csv(INPUT_FILE)

        # Train, Predict, and Evaluate
        train_and_evaluate_xg_model(raw_df)
