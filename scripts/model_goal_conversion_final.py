# model_goal_conversion_final.py
# GOAL: Final stabilization using Logistic Regression with L1 (Lasso) Regularization
# to enforce generalization and reach CV ROC AUC score > 0.65.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
# --- SWITCHING BACK TO LOGISTIC REGRESSION WITH LASSO (L1) ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Any

# Set the base directory (TEST/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"


# --- Elo Rating Calculation Function (Copied for completeness) ---

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


# --- Step 2: Modeling and Evaluation (Switching to Logistic Regression with L1) ---

def train_and_evaluate_goal_conversion_model(df: pd.DataFrame):
    """
    Trains a stable Logistic Regression model with L1 Regularization (Lasso)
    for Goal vs. No Goal.
    """

    # --- APPLY FILTER ---
    # Filter for known strikers and known ELO (non-1500 default)
    df_model = df[(df['striker_elo_prev'] != 1500)].copy()

    # === CRITICAL FIX: FEATURE ENGINEERING ELO ADVANTAGE ===
    # The ELO system is based on the difference, not absolute values.
    df_model['elo_advantage'] = df_model['striker_elo_prev'] - df_model['keeper_elo_prev']
    # =======================================================

    # Define Features (X) - LEAN GOAL CONVERSION SET
    features = [
        'footedness',
        'minute',
        'match_scoreline_diff',
        'elo_advantage',
        'home_or_away'
    ]

    # TARGET DEFINITION: Binary Classification (0=Not Goal, 1=Goal)
    target = 'penalty_outcome'
    df_model['target'] = (df_model[target] == 'Goal').astype(int)

    X = df_model[features]
    y_encoded = df_model['target']

    # Class 0: 'Not Goal', Class 1: 'Goal'
    target_names = ['Not Goal', 'Goal']

    print(f"\nTotal model samples (filtered for Goal Conversion): {len(X)} samples")
    print(f"Target Labels (0, 1): {target_names}")

    # --- Create Preprocessing Pipeline ---
    categorical_features = ['footedness', 'home_or_away']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with L1 Logistic Regression ---

    # L1 (Lasso) Regularization forces unused feature coefficients to 0,
    # stabilizing the model and enforcing generalization on small data.
    classifier = LogisticRegression(
        penalty='l1',  # Use L1 (Lasso) regularization for stability
        solver='liblinear',  # Compatible solver for L1
        C=0.25,  # FINAL TWEAK: Increased regularization strength (C=0.5 -> C=0.25)
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

    print(f"\n--- Training Goal Conversion Model (Logistic Regression L1 WEIGHTED) ---")

    # Train the final model on all data
    model_pipeline.fit(X, y_encoded)
    best_model = model_pipeline

    # Calculate Cross-Validation Score (ROC AUC is the preferred metric for imbalance)
    cv_scores = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_score = np.mean(cv_scores)

    # --- Feature Importance (Coefficients) Analysis for Logistic Regression L1 ---
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

    # Get coefficients (weights) from the fitted Logistic Regression model
    coefficients = best_model.named_steps['classifier'].coef_[0]

    importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Importance in LR is measured by the absolute value of the coefficient
    importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
    importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])

    print("\n--- Top Feature Importances (Logistic Regression L1 Coefficients) ---")
    print("NOTE: Coefficient sign indicates positive/negative correlation with 'Goal' (1).")
    print("Features with coefficients near zero have been effectively removed by L1 penalty.")
    print(importance_df.to_string(index=False))

    # --- Evaluation ---
    y_pred_encoded = best_model.predict(X)

    print("\n--- Model Evaluation (Optimized Logistic Regression L1 WEIGHTED) ---")

    # 1. Overall Accuracy on ALL data
    accuracy = accuracy_score(y_encoded, y_pred_encoded)
    print(f"Model Accuracy on Full Dataset ({len(X)} samples): {accuracy:.4f}")

    # 2. Detailed Classification Report
    print("\nClassification Report (Full Dataset):")
    print(classification_report(y_encoded, y_pred_encoded, target_names=target_names, zero_division=0))

    # 3. K-Fold Cross-Validated Score (Primary CV metric)
    print(f"Cross-Validated ROC AUC Score (5-Fold Mean): {cv_score:.4f} (Targeting > 0.65 for stability)")

    # 4. Confusion Matrix
    conf_mat = confusion_matrix(y_encoded, y_pred_encoded)

    print("\nConfusion Matrix (Rows=Actual, Columns=Predicted - Full Dataset):")
    print(pd.DataFrame(conf_mat,
                       index=[f'Actual {l}' for l in target_names],
                       columns=[f'Predicted {l}' for l in target_names]))

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------
    MODEL_PATH = OUTPUT_DIR / "goal_conversion_final_model.pkl"
    joblib.dump(best_model, MODEL_PATH)

    print("\n--- Model Persistence ---")
    print(f"Goal Conversion Model Pipeline saved to: {MODEL_PATH}")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print(
        "CORE MODEL 1: Logistic Regression BINARY CLASSIFICATION (Goal vs Not Goal) with L1 REGULARIZATION for superior stability.")


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run 'scripts/main.py' first to generate the 'penalty_event_log.csv'.")
    else:
        # Load the data
        raw_df = pd.read_csv(INPUT_FILE)

        # Step 1: Feature Engineering (only need ELO)
        print("Starting ELO rating feature engineering...")
        engineered_df = calculate_elo_rating(raw_df)

        # Step 2: Train, Predict, and Evaluate
        train_and_evaluate_goal_conversion_model(engineered_df)
