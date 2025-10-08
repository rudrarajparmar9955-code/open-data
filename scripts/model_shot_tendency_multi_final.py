# model_shot_tendency_multi_final.py
# GOAL: Build the final Multiclass Classification model (Left vs Right vs Center).
# 1. Use the Minimal feature set plus historical percentages for all 3 sides (Total 9 features).
# 2. Implement class weighting to handle the severe imbalance (Center is the majority class).
# 3. Use XGBoost's 'multi:softmax' objective.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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
    for the keeper, and includes Elo ratings.
    """
    KEEPER_COLUMN = 'goalkeeper_name'

    df = calculate_elo_rating(df, default_elo=1500, k_factor=32)
    df = df.sort_values(by=['striker', 'match_date', 'minute', 'second'])

    # Striker Tendencies
    df['cum_total_attempts'] = df.groupby('striker')['match_id'].cumcount() + 1
    df['cum_total_prev'] = df.groupby('striker')['cum_total_attempts'].shift(1, fill_value=0)

    def safe_div(numerator, denominator):
        # Calculates percentage, handling division by zero for the first shot
        return np.where(denominator > 0, numerator / denominator, 0)

    for side in ['Left', 'Right', 'Center']:
        side_lower = side.lower()
        # 1. Flag shot side
        df[f'is_{side_lower}'] = (df['shot_side'] == side).astype(int)
        # 2. Calculate cumulative counts
        df[f'cum_{side_lower}'] = df.groupby('striker')[f'is_{side_lower}'].cumsum()
        # 3. Shift to get previous cumulative counts
        df[f'cum_{side_lower}_prev'] = df.groupby('striker')[f'cum_{side_lower}'].shift(1, fill_value=0)
        # 4. Calculate historical percentage
        df[f'hist_{side_lower}_pct'] = safe_div(df[f'cum_{side_lower}_prev'], df['cum_total_prev'])

    # Clean up intermediate columns (except *_prev and *_pct features)
    cols_to_drop = [col for col in df.columns if col.startswith(('cum_', 'is_', 'keeper_cum_')) and not col.endswith(
        '_prev')]
    cols_to_drop.extend(['outcome'])
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


# --- Step 2: Modeling and Evaluation ---

def train_and_evaluate_multiclass_model(df: pd.DataFrame):
    """
    Trains a MINIMAL multiclass classification model (Left vs Right vs Center)
    using key features and applies sample weighting to balance recall.
    """

    # --- APPLY FILTER ---
    # Filter for known strikers and known ELO (non-1500 default)
    df_model = df[(df['cum_total_prev'] > 0) & (df['striker_elo_prev'] != 1500)].copy()

    # Define Features (X) - MINIMAL MULTICLASS FEATURE SET (6 Total - Removed hist_% to prevent leakage)
    features = [
        'footedness',
        'minute',
        'match_scoreline_diff',
        'cum_total_prev',
        'keeper_elo_prev',
        'striker_elo_prev',  # Added striker ELO back as it's a critical feature for context
    ]

    # TARGET DEFINITION: Multiclass Classification (Left, Right, Center)
    target = 'shot_side'
    valid_sides = ['Left', 'Right', 'Center']
    # Filter only for the classes we intend to model (should be all non-Unknown)
    df_model = df_model[df_model[target].isin(valid_sides)].copy()

    X = df_model[features]
    y_str = df_model[target]

    # Convert string labels to integers for XGBoost (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    # This will determine the order: 0='Center', 1='Left', 2='Right' based on alphabetical order.
    target_names = le.classes_
    num_classes = len(target_names)

    # --- CALCULATE CLASS WEIGHTS ---
    # Multiclass requires computing weights for all classes to create sample weights.
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_map = {i: weights[i] for i in np.unique(y_encoded)}

    # Create sample weights for the entire dataset
    sample_weights = np.array([class_weight_map[label] for label in y_encoded])

    print(f"\nTotal model samples (filtered for Multiclass L/R/C): {len(X)} samples")
    print(f"Class counts: {y_str.value_counts().to_dict()}")
    print(f"Target Labels (0, 1, 2): {target_names}")
    print(f"Calculated Class Weights: {class_weight_map}")

    # --- Create Preprocessing Pipeline ---
    categorical_features = ['footedness']
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # --- Create Modeling Pipeline with RandomizedSearchCV ---

    # Multiclass classification objective
    classifier = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
        # Note: Class weights are passed via the 'sample_weight' parameter in .fit()
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

    # StratifiedKFold for multiclass
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=100,
        cv=cv,
        scoring='accuracy',  # Use accuracy for overall performance
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    # Train the model, passing sample weights to the fit method
    print(f"\n--- Training Minimal Multiclass Shot Tendency Model (XGBoost WEIGHTED L/R/C) ---")
    random_search.fit(X, y_encoded, classifier__sample_weight=sample_weights)

    # Get the best model
    best_model = random_search.best_estimator_
    cv_accuracy_score = random_search.best_score_

    print("\n--- Best Hyperparameters Found (RandomizedSearchCV) ---")
    best_params_clean = {k.split('__')[1]: v for k, v in random_search.best_params_.items() if
                         k.startswith('classifier__')}
    print(best_params_clean)

    # --- Feature Importance Analysis ---
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\n--- Top 9 Feature Importances (Best MULTICLASS L/R/C Model) ---")
    print(importance_df.head(9).to_string(index=False))

    # --- Evaluation ---
    y_pred_encoded = best_model.predict(X)
    y_pred_str = le.inverse_transform(y_pred_encoded)

    print("\n--- Model Evaluation (Optimized XGBoost MINIMAL WEIGHTED Multiclass L/R/C) ---")

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
    MODEL_PATH = OUTPUT_DIR / "shot_tendency_multi_final_model.pkl"
    joblib.dump(best_model, MODEL_PATH)

    print("\n--- Model Persistence ---")
    print(f"Multiclass Weighted L/R/C Model Pipeline saved to: {MODEL_PATH}")

    # --- Portfolio Insight ---
    print("\n--- Portfolio Insight ---")
    print(
        "FINAL MODEL: XGBoost MULTICLASS CLASSIFICATION (Left vs Right vs Center) with MINIMAL feature set and CLASS WEIGHTING.")


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
        train_and_evaluate_multiclass_model(engineered_df)
