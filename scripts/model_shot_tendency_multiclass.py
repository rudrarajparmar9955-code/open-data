# model_shot_tendency_binary.py
# GOAL: Shifts focus to a cleaner **BINARY CLASSIFICATION** problem (Left vs. Right).
# This drops the dominant 'Center' class to combat overfitting and improve generalization
# on the truly discriminative task.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import joblib

# Set the base directory (TEST/) by going up one level from the scripts directory
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"


# --- Elo Rating Calculation Function (Copied from previous file) ---

def calculate_elo_rating(df: pd.DataFrame, default_elo=1500, k_factor=32) -> pd.DataFrame:
    """Calculates dynamic Elo ratings for strikers and keepers."""
    KEEPER_COLUMN = 'goalkeeper_name'
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

        E_striker = 1 / (1 + 10 ** ((keeper_elo - striker_elo) / 400))

        striker_new_elo = striker_elo + k_factor * (outcome - E_striker)
        keeper_new_elo = keeper_elo + k_factor * ((1 - outcome) - (1 - E_striker))

        striker_elos[striker] = striker_new_elo
        keeper_elos[keeper] = keeper_new_elo

    elo_df = pd.DataFrame({
        'striker_elo_prev': striker_elo_results,
        'keeper_elo_prev': keeper_elo_results
    })
    return df.join(elo_df)


# --- Step 1: Feature Engineering (Cumulative Historical Tendency) (Copied from previous file) ---

def engineer_historical_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cumulative historical shot tendencies for the striker AND save tendencies
    for the keeper, and includes Elo ratings.
    """
    PENALTY_RESULT_COLUMN = 'penalty_outcome'
    KEEPER_COLUMN = 'goalkeeper_name'

    # 0. Calculate ELO ratings
    df = calculate_elo_rating(df, default_elo=1500, k_factor=32)

    # Ensure data is sorted chronologically
    df = df.sort_values(by=['striker', 'match_date', 'minute', 'second'])

    # --- 1. Striker Shot Tendencies (Direction) ---
    df['is_left'] = (df['shot_side'] == 'Left').astype(int)
    df['is_right'] = (df['shot_side'] == 'Right').astype(int)
    df['is_center'] = (df['shot_side'] == 'Center').astype(int)

    df['cum_left'] = df.groupby('striker')['is_left'].cumsum()
    df['cum_right'] = df.groupby('striker')['is_right'].cumsum()
    df['cum_center'] = df.groupby('striker')['is_center'].cumsum()

    df['cum_total_attempts'] = df.groupby('striker')['match_id'].cumcount() + 1

    # Shift counts down by 1 to represent data *before* the current penalty
    df['cum_left_prev'] = df.groupby('striker')['cum_left'].shift(1, fill_value=0)
    df['cum_right_prev'] = df.groupby('striker')['cum_right'].shift(1, fill_value=0)
    df['cum_center_prev'] = df.groupby('striker')['cum_center'].shift(1, fill_value=0)
    df['cum_total_prev'] = df.groupby('striker')['cum_total_attempts'].shift(1, fill_value=0)

    # Calculate Historical Percentages (Tendencies)
    def safe_div(numerator, denominator):
        """Avoids division by zero when a player has no prior attempts by returning 0."""
        return np.where(denominator > 0, numerator / denominator, 0)

    df['hist_left_pct'] = safe_div(df['cum_left_prev'], df['cum_total_prev'])
    df['hist_right_pct'] = safe_div(df['cum_right_prev'], df['cum_total_prev'])
    df['hist_center_pct'] = safe_div(df['cum_center_prev'], df['cum_total_prev'])

    # --- 3. Keeper Historical Tendencies (Save/Miss) ---

    # Sort by keeper now to calculate keeper cumulative stats
    df = df.sort_values(by=[KEEPER_COLUMN, 'match_date', 'minute', 'second'])

    # Determine if the keeper succeeded (save or miss) for each shot side
    df['is_left_save'] = ((df['shot_side'] == 'Left') & (df[PENALTY_RESULT_COLUMN].isin(['Saved', 'Missed']))).astype(
        int)
    df['is_right_save'] = ((df['shot_side'] == 'Right') & (df[PENALTY_RESULT_COLUMN].isin(['Saved', 'Missed']))).astype(
        int)
    df['is_center_save'] = (
            (df['shot_side'] == 'Center') & (df[PENALTY_RESULT_COLUMN].isin(['Saved', 'Missed']))).astype(int)

    # Cumulative count of times keeper faced a shot to that side
    df['keeper_cum_left_faced'] = (df['shot_side'] == 'Left').astype(int).groupby(df[KEEPER_COLUMN]).cumsum()
    df['keeper_cum_right_faced'] = (df['shot_side'] == 'Right').astype(int).groupby(df[KEEPER_COLUMN]).cumsum()
    df['keeper_cum_center_faced'] = (df['shot_side'] == 'Center').astype(int).groupby(df[KEEPER_COLUMN]).cumsum()

    # Cumulative saves/misses for each side
    df['keeper_cum_left_save'] = df.groupby(KEEPER_COLUMN)['is_left_save'].cumsum()
    df['keeper_cum_right_save'] = df.groupby(KEEPER_COLUMN)['is_right_save'].cumsum()
    df['keeper_cum_center_save'] = df.groupby(KEEPER_COLUMN)['is_center_save'].cumsum()

    # Total penalties faced by keeper
    df['keeper_cum_total_faced'] = df.groupby(KEEPER_COLUMN)['match_id'].cumcount() + 1

    # Shift counts down by 1 (PREV state)
    for col in ['keeper_cum_left_save', 'keeper_cum_right_save', 'keeper_cum_center_save',
                'keeper_cum_left_faced', 'keeper_cum_right_faced', 'keeper_cum_center_faced',
                'keeper_cum_total_faced']:
        df[f'{col}_prev'] = df.groupby(KEEPER_COLUMN)[col].shift(1, fill_value=0)

    # New Keeper Tendency Features (Percentage of success facing a shot to that side)
    df['keeper_hist_left_save_pct'] = safe_div(df['keeper_cum_left_save_prev'], df['keeper_cum_left_faced_prev'])
    df['keeper_hist_right_save_pct'] = safe_div(df['keeper_cum_right_save_prev'], df['keeper_cum_right_faced_prev'])
    df['keeper_hist_center_save_pct'] = safe_div(df['keeper_cum_center_save_prev'], df['keeper_cum_center_faced_prev'])
    df['keeper_total_faced_prev'] = df['keeper_cum_total_faced_prev']

    # Drop intermediate and unneeded columns
    cols_to_drop = [col for col in df.columns if col.startswith(('cum_', 'is_', 'keeper_cum_')) and not col.endswith(
        '_prev') and col != 'keeper_total_faced_prev']
    cols_to_drop.extend(['outcome'])
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


# --- Step 2: Modeling and Evaluation ---

def train_and_evaluate_binary_model(df: pd.DataFrame):
    """
    Trains a BINARY classification model (Left vs Right) on the filtered dataset.
    We drop the 'Center' class and remove class weighting.
    """

    # --- APPLY FILTER ---
    df_model = df[(df['cum_total_prev'] > 0) & (df['striker_elo_prev'] != 1500)].copy()

    # Define Features (X) - SAME FEATURES AS BEFORE
    features = [
        'footedness', 'home_or_away', 'match_scoreline_diff', 'minute',

        # Striker Tendency Features
        'hist_left_pct', 'hist_right_pct', 'hist_center_pct', 'cum_total_prev',

        # Striker and Keeper ELO
        'striker_elo_prev',
        'keeper_elo_prev',

        # Keeper Tendency Features
        'keeper_hist_left_save_pct', 'keeper_hist_right_save_pct',
        'keeper_hist_center_save_pct', 'keeper_total_faced_prev'
    ]

    # TARGET DEFINITION: Binary Classification (Left vs Right ONLY)
    target = 'shot_side'
    valid_sides = ['Left', 'Right']  # <<< KEY CHANGE: Only focus on these two
    df_model = df_model[df_model[target].isin(valid_sides)].copy()

    X = df_model[features]
    y_str = df_model[target]

    # Convert string labels to integers for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    target_names = le.classes_

    # --- NO CLASS WEIGHTING NEEDED ---
    # The new binary classes are 109 Left, 110 Right, which is balanced.

    print(f"\nTotal model samples (filtered for Binary L/R): {len(X)} samples")
    print(f"Using {len(X)} samples for 5-Fold Cross-Validation tuning.")
    print(f"Class balance (Binary Left vs Right):\n{y_str.value_counts(normalize=True)}")

    # --- Create Preprocessing Pipeline (No change) ---
    categorical_features = ['footedness', 'home_or_away']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with RandomizedSearchCV ---

    # Binary classification objective
    classifier = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Using the regularized search space, but slightly wider bounds for depth/estimators now
    param_distributions = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.6, 0.4),
        'classifier__gamma': uniform(0, 0.5),
        'classifier__reg_alpha': uniform(0.0, 5.0),  # L1 regularization
        'classifier__reg_lambda': uniform(0.0, 5.0)  # L2 regularization
    }

    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=100,
        cv=cv,
        scoring='accuracy',  # Accuracy is fine since classes are balanced
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    # Train the model (WITHOUT SAMPLE WEIGHTS)
    print(f"\n--- Training Shot Tendency Model (XGBoost BINARY L/R) using 5-Fold Stratified CV ---")
    random_search.fit(X, y_encoded)  # No sample_weight argument passed

    # Get the best model
    best_model = random_search.best_estimator_

    print("\n--- Best Hyperparameters Found (RandomizedSearchCV) ---")
    best_params_clean = {k.split('__')[1]: v for k, v in random_search.best_params_.items() if
                         k.startswith('classifier__')}
    print(best_params_clean)

    cv_accuracy_score = random_search.best_score_

    # --- Feature Importance Analysis ---
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = best_model.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\n--- Top 10 Feature Importances (Best Binary L/R Model) ---")
    print(importance_df.head(10).to_string(index=False))

    # --- Evaluation ---

    y_pred_encoded = best_model.predict(X)
    y_pred_str = le.inverse_transform(y_pred_encoded)

    print("\n--- Model Evaluation (Optimized XGBoost Binary L/R) ---")

    # 1. Overall Accuracy on ALL data
    accuracy = accuracy_score(y_str, y_pred_str)
    print(f"Model Accuracy on Full Dataset ({len(X)} samples): {accuracy:.4f}")

    # 2. Detailed Classification Report
    print("\nClassification Report (Full Dataset):")
    print(classification_report(y_str, y_pred_str, target_names=target_names, zero_division=0))

    # 3. K-Fold Cross-Validated Score (Primary CV metric)
    print(f"Cross-Validated Accuracy Score (5-Fold Mean): {cv_accuracy_score:.4f} (Closer to 1.0 is better)")

    # 4. Confusion Matrix
    conf_mat = confusion_matrix(y_str, y_pred_str, labels=target_names)

    print("\nConfusion Matrix (Rows=Actual, Columns=Predicted - Full Dataset):")
    print(pd.DataFrame(conf_mat,
                       index=[f'Actual {l}' for l in target_names],
                       columns=[f'Predicted {l}' for l in target_names]))

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------
    MODEL_PATH = OUTPUT_DIR / "shot_tendency_binary_model.pkl"
    joblib.dump(best_model, MODEL_PATH)

    print("\n--- Model Persistence ---")
    print(f"Binary Left/Right Model Pipeline saved to: {MODEL_PATH}")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print("FINAL MODEL: XGBoost BINARY CLASSIFICATION (Left vs Right ONLY) tuned using ACCURACY.")


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run 'main.py' first to generate the 'penalty_event_log.csv'.")
    else:
        # Load the data
        raw_df = pd.read_csv(INPUT_FILE)

        # Step 1: Feature Engineering
        print("Starting historical feature engineering...")
        engineered_df = engineer_historical_tendencies(raw_df)

        # Step 2 & 3 & 4: Train, Predict, and Evaluate
        train_and_evaluate_binary_model(engineered_df)
