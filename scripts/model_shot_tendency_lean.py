# model_shot_tendency_minimal.py
# GOAL: Create the final, leanest model by:
# 1. Dropping the zero-importance 'home_or_away' feature (Total 7 features).
# 2. Introducing a slight class weight adjustment (scale_pos_weight) to combat the observed bias
#    towards predicting 'Right' and improve 'Left' recall (currently 49%).

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

# Set the base directory (TEST/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"


# --- Elo Rating Calculation Function (Copied for completeness) ---

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


# --- Feature Engineering (Cumulative Historical Tendency) (Copied for completeness) ---

def engineer_historical_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cumulative historical shot tendencies for the striker AND save tendencies
    for the keeper, and includes Elo ratings. This function is left comprehensive
    even though many of the generated features will be dropped later.
    """
    KEEPER_COLUMN = 'goalkeeper_name'
    PENALTY_RESULT_COLUMN = 'penalty_outcome'

    df = calculate_elo_rating(df, default_elo=1500, k_factor=32)
    df = df.sort_values(by=['striker', 'match_date', 'minute', 'second'])

    # Striker Tendencies
    df['is_left'] = (df['shot_side'] == 'Left').astype(int)
    df['cum_left'] = df.groupby('striker')['is_left'].cumsum()
    df['cum_total_attempts'] = df.groupby('striker')['match_id'].cumcount() + 1
    df['cum_left_prev'] = df.groupby('striker')['cum_left'].shift(1, fill_value=0)
    df['cum_total_prev'] = df.groupby('striker')['cum_total_attempts'].shift(1, fill_value=0)

    def safe_div(numerator, denominator):
        return np.where(denominator > 0, numerator / denominator, 0)

    df['hist_left_pct'] = safe_div(df['cum_left_prev'], df['cum_total_prev'])

    # We also need the other tendency features to be calculated but they are dropped in the feature list
    for side in ['right', 'center']:
        df[f'is_{side}'] = (df['shot_side'] == side.capitalize()).astype(int)
        df[f'cum_{side}'] = df.groupby('striker')[f'is_{side}'].cumsum()
        df[f'cum_{side}_prev'] = df.groupby('striker')[f'cum_{side}'].shift(1, fill_value=0)
        df[f'hist_{side}_pct'] = safe_div(df[f'cum_{side}_prev'], df['cum_total_prev'])

    # Keeper Tendencies (needed for consistency, even if not used as features)
    df = df.sort_values(by=[KEEPER_COLUMN, 'match_date', 'minute', 'second'])
    df['keeper_cum_total_faced'] = df.groupby(KEEPER_COLUMN)['match_id'].cumcount() + 1
    df['keeper_cum_total_faced_prev'] = df.groupby(KEEPER_COLUMN)['keeper_cum_total_faced'].shift(1, fill_value=0)

    cols_to_drop = [col for col in df.columns if col.startswith(('cum_', 'is_', 'keeper_cum_')) and not col.endswith(
        '_prev') and col != 'keeper_cum_total_faced_prev']
    cols_to_drop.extend(['outcome'])
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


# --- Step 2: Modeling and Evaluation ---

def train_and_evaluate_minimal_model(df: pd.DataFrame):
    """
    Trains a MINIMAL binary classification model (Left vs Right) using 7 key features
    and applies a slight class weight to balance recall.
    """

    # --- APPLY FILTER ---
    df_model = df[(df['cum_total_prev'] > 0) & (df['striker_elo_prev'] != 1500)].copy()

    # Define Features (X) - MINIMAL FEATURE SET (7 Features)
    features = [
        'footedness',
        'minute',
        'match_scoreline_diff',
        'hist_left_pct',
        'cum_total_prev',
        'keeper_elo_prev',
    ]

    # TARGET DEFINITION: Binary Classification (Left vs Right ONLY)
    target = 'shot_side'
    valid_sides = ['Left', 'Right']
    df_model = df_model[df_model[target].isin(valid_sides)].copy()

    X = df_model[features]
    y_str = df_model[target]

    # Convert string labels to integers for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    target_names = le.classes_  # 0='Left', 1='Right'

    # --- NEW: CALCULATE CLASS WEIGHT ---
    # We want to increase the penalty for misclassifying the MINORITY/HARDER class (Left = 0).
    # XGBoost uses scale_pos_weight which is (Count of Negative Class) / (Count of Positive Class).
    # Since 'Right' is label 1 (positive class) and 'Left' is label 0 (negative class),
    # and the number of samples are ~equal, we apply a small factor > 1 to bias the model to 'Left'.
    # We will swap the labels so that 'Left' is the positive class (1) for weighting purposes.

    # 0 = Right, 1 = Left
    y_str_reverted = np.where(y_str == 'Left', '1', '0')
    y_encoded_reverted = y_str_reverted.astype(int)

    count_left = (y_str == 'Left').sum()
    count_right = (y_str == 'Right').sum()

    # The ideal ratio is 1.01 (110 Right / 109 Left). We apply a small bump (1.2) to push the model.
    # We set 'Left' as the positive class (1) for weight calculation.
    # Weight = (Count of Negative Class) / (Count of Positive Class) = Count_Right / Count_Left
    weight = count_right / count_left
    # Apply a slight additional boost (1.2x) to 'Left' recall as observed from confusion matrix
    scale_pos_weight_value = weight * 1.2

    print(f"\nTotal model samples (filtered for Minimal L/R): {len(X)} samples")
    print(f"Class balance: Left={count_left}, Right={count_right}. Scale_pos_weight used: {scale_pos_weight_value:.4f}")

    # --- Create Preprocessing Pipeline ---
    categorical_features = ['footedness']  # Only footedness is categorical now
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with RandomizedSearchCV ---

    # Binary classification objective, using the calculated weight
    classifier = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight_value,  # <<< NEW: Class Weight Added
        random_state=42
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    param_distributions = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.6, 0.4),
        'classifier__gamma': uniform(0, 0.5),
        'classifier__reg_alpha': uniform(0.0, 5.0),
        'classifier__reg_lambda': uniform(0.0, 5.0)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=100,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    # Train the model using the reverted (weighted) labels: 0=Right, 1=Left
    print(f"\n--- Training Minimal Shot Tendency Model (XGBoost WEIGHTED BINARY L/R) ---")
    random_search.fit(X, y_encoded_reverted)

    # Get the best model
    best_model = random_search.best_estimator_
    cv_accuracy_score = random_search.best_score_

    print("\n--- Best Hyperparameters Found (RandomizedSearchCV) ---")
    best_params_clean = {k.split('__')[1]: v for k, v in random_search.best_params_.items() if
                         k.startswith('classifier__')}
    print(best_params_clean)

    # --- Feature Importance Analysis (using original features for clarity) ---
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\n--- Top 7 Feature Importances (Best MINIMAL Binary L/R Model) ---")
    print(importance_df.head(7).to_string(index=False))

    # --- Evaluation ---

    # Predict and revert labels back to original ('Left', 'Right') for reporting
    y_pred_encoded_reverted = best_model.predict(X)
    y_pred_str = np.where(y_pred_encoded_reverted == 1, 'Left', 'Right')

    print("\n--- Model Evaluation (Optimized XGBoost MINIMAL WEIGHTED Binary L/R) ---")

    # 1. Overall Accuracy on ALL data
    accuracy = accuracy_score(y_str, y_pred_str)
    print(f"Model Accuracy on Full Dataset ({len(X)} samples): {accuracy:.4f}")

    # 2. Detailed Classification Report
    print("\nClassification Report (Full Dataset):")
    # Need to pass y_str and y_pred_str (original labels) to classification_report
    print(classification_report(y_str, y_pred_str, target_names=target_names, zero_division=0))

    # 3. K-Fold Cross-Validated Score (Primary CV metric)
    print(f"Cross-Validated Accuracy Score (5-Fold Mean): {cv_accuracy_score:.4f} (Closer to 1.0 is better)")

    # 4. Confusion Matrix
    conf_mat = confusion_matrix(y_str, y_pred_str, labels=target_names)  # labels=['Left', 'Right']

    print("\nConfusion Matrix (Rows=Actual, Columns=Predicted - Full Dataset):")
    print(pd.DataFrame(conf_mat,
                       index=[f'Actual {l}' for l in target_names],
                       columns=[f'Predicted {l}' for l in target_names]))

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------
    MODEL_PATH = OUTPUT_DIR / "shot_tendency_minimal_weighted_model.pkl"
    joblib.dump(best_model, MODEL_PATH)

    print("\n--- Model Persistence ---")
    print(f"Minimal Weighted Binary L/R Model Pipeline saved to: {MODEL_PATH}")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print(
        "FINAL MODEL: XGBoost BINARY CLASSIFICATION (Left vs Right ONLY) with MINIMAL feature set and CLASS WEIGHTING.")


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run 'scripts/main.py' first to generate the 'penalty_event_log.csv'.")
    else:
        # Load the data
        raw_df = pd.read_csv(INPUT_FILE)

        # Step 1: Feature Engineering
        print("Starting historical feature engineering...")
        engineered_df = engineer_historical_tendencies(raw_df)

        # Step 2 & 3 & 4: Train, Predict, and Evaluate
        train_and_evaluate_minimal_model(engineered_df)
