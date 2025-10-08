# model_shot_tendency.py
# Trains a model to predict the striker's shot direction (Left vs. Not Left)
# using only pre-shot features, incorporating Keeper and Striker Elo Skill Proxies.
# CHANGE: Simplified the target variable to a Binary Classification problem (Left vs. Not Left).

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Import XGBoost for maximum performance uplift
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set the base directory (TEST/) by going up one level from the scripts directory
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"


# --- Elo Rating Calculation Function (UNCHANGED) ---

def calculate_elo_rating(df: pd.DataFrame, default_elo=1500, k_factor=32) -> pd.DataFrame:
    """
    Calculates dynamic Elo ratings for strikers and keepers over time,
    based on penalty outcomes.
    """
    KEEPER_COLUMN = 'goalkeeper_name'

    # Ensure data is sorted chronologically for correct time-series calculation
    df = df.sort_values(by=['match_date', 'minute', 'second']).reset_index(drop=True)

    # Initialize Elo scores dictionaries
    striker_elos = {}
    keeper_elos = {}

    # Store results (these will be the features used for prediction, representing PREV state)
    striker_elo_results = []
    keeper_elo_results = []

    # Outcome: 1.0 if GOAL (striker win), 0.0 otherwise (keeper win/save/miss)
    df['outcome'] = (df['penalty_outcome'] == 'Goal').astype(float)

    for index, row in df.iterrows():
        striker = row['striker']
        keeper = row[KEEPER_COLUMN]
        outcome = row['outcome']

        # Get current ELO for striker and keeper, defaulting to 1500 if new
        striker_elo = striker_elos.get(striker, default_elo)
        keeper_elo = keeper_elos.get(keeper, default_elo)

        # Record the ELO *before* the penalty is taken (the PREV state)
        striker_elo_results.append(striker_elo)
        keeper_elo_results.append(keeper_elo)

        # Calculate expected outcome (E) for the striker (probability of scoring)
        E_striker = 1 / (1 + 10 ** ((keeper_elo - striker_elo) / 400))

        # Calculate new ELOs
        striker_new_elo = striker_elo + k_factor * (outcome - E_striker)
        # Keeper's score is (1 - outcome). Keeper's expected score is (1 - E_striker).
        keeper_new_elo = keeper_elo + k_factor * ((1 - outcome) - (1 - E_striker))

        # Update dictionaries for the next iteration
        striker_elos[striker] = striker_new_elo
        keeper_elos[keeper] = keeper_new_elo

    # Combine results into a DataFrame
    elo_df = pd.DataFrame({
        'striker_elo_prev': striker_elo_results,
        'keeper_elo_prev': keeper_elo_results
    })

    # Merge Elo results back into the main DataFrame
    return df.join(elo_df)


# --- Step 1: Feature Engineering (Cumulative Historical Tendency) ---

def engineer_historical_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cumulative historical shot tendencies for the striker AND save tendencies
    for the keeper, and includes Elo ratings. (UNCHANGED logic for features)
    """

    # === COLUMN DEFINITIONS BASED ON USER INPUT ===
    PENALTY_RESULT_COLUMN = 'penalty_outcome'
    KEEPER_COLUMN = 'goalkeeper_name'
    # ==========================================

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
        """Avoids division by zero when a player has no prior attempts."""
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

def train_and_evaluate_tendency_model(df: pd.DataFrame):
    """
    Trains a BINARY classification model to predict if the shot is Left or Not Left.
    """

    # Filter out penalties where the striker has no prior history or the Elo is default
    df_model = df[(df['cum_total_prev'] > 0) & (df['striker_elo_prev'] != 1500)].copy()

    # --- Confirmation: Sample of Elo Features in Training Data (Top 5 Rows) ---
    print("\n--- Confirmation: Sample of Elo Features in Training Data (Top 5 Rows) ---")
    elo_cols = ['striker', 'goalkeeper_name', 'striker_elo_prev', 'keeper_elo_prev', 'shot_side']
    existing_elo_cols = [c for c in elo_cols if c in df_model.columns]
    print(df_model[existing_elo_cols].head().to_string(index=False))
    print("-" * 75)

    # ----------------------------------------------------
    # Define Features (X) (UNCHANGED)
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # NEW TARGET DEFINITION: Binary Classification
    # ----------------------------------------------------
    target = 'is_left_shot'
    # 1 if shot is Left, 0 if shot is Right or Center
    df_model[target] = (df_model['shot_side'] == 'Left').astype(int)

    X = df_model[features]
    y = df_model[target]

    # Drop rows where shot_side was missing or unclassified (if any, though not expected here)
    y = y[df_model['shot_side'].isin(['Left', 'Right', 'Center'])]
    X = X.loc[y.index]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Test set class balance (Left=1, Not Left=0):\n{y_test.value_counts(normalize=True)}")

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

    # --- Create Modeling Pipeline with RandomizedSearchCV ---

    # Define the XGBoost Classifier for Binary Classification
    classifier = XGBClassifier(
        objective='binary:logistic',  # CHANGE: Binary objective
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # Initialize SMOTE for oversampling (still good practice for binary imbalance)
    smote = SMOTE(random_state=42)

    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', classifier)
    ])

    # Define the parameter distribution for RandomizedSearchCV (adjusted for binary)
    param_distributions = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 10),  # Slightly reduced max_depth for binary
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.6, 0.4),
        'classifier__gamma': uniform(0, 0.5),
        # Scale weight of positive examples (Left shot = 1)
        'classifier__scale_pos_weight': [1, 1.5, 2]  # Experimenting with manual weighting
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='f1',  # Use standard F1 for binary classification
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    # Train the model using Randomized Search
    print(f"\n--- Training Shot Tendency Model (XGBoost Binary Left vs. Not Left) ---")
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    print("\n--- Best Hyperparameters Found (RandomizedSearchCV) ---")
    best_params_clean = {k.split('__')[1]: v for k, v in random_search.best_params_.items() if
                         k.startswith('classifier__')}
    print(best_params_clean)

    # --- Feature Importance Analysis (UNCHANGED Logic) ---

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = best_model.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\n--- Top 10 Feature Importances (Best Model) ---")
    print(importance_df.head(10).to_string(index=False))

    # --- Evaluation ---

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of "Left" (class 1)

    print("\n--- Model Evaluation (Optimized XGBoost Binary: Left vs. Not Left) ---")

    # 1. Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # 2. Detailed Classification Report
    target_names = ['Not Left (0)', 'Left (1)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 3. AUC-ROC (Key metric for binary classification)
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC Score: {auc_roc:.4f} (Closer to 1.0 is better)")
    except ValueError:
        print("AUC-ROC could not be calculated (requires probability scores).")

    # 4. Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix (Rows=Actual, Columns=Predicted):")
    print(pd.DataFrame(conf_mat,
                       index=[f'Actual {l}' for l in target_names],
                       columns=[f'Predicted {l}' for l in target_names]))

    # Interpretation for the portfolio:
    print("\n--- Portfolio Insight ---")
    print("Interpretation: The Diagonal of the Confusion Matrix (top-left to bottom-right) shows")
    print("how often the model correctly predicted the shot side (Left or Not Left). The AUC-ROC is")
    print("the overall measure of the model's ability to distinguish between the two classes.")
    print(
        f"\nIMPROVEMENT APPLIED: Switched the problem to a BINARY classification (Left vs. Not Left) to leverage the majority class and stabilize predictions. Best F1-score: {random_search.best_score_:.4f}")


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
        train_and_evaluate_tendency_model(engineered_df)
