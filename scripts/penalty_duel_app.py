# penalty_duel_app.py
# The 12-Yard Matchup Predictor: A Streamlit web application for real-time penalty analysis.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path
from typing import Dict, Any, List

# --- Configuration & Paths ---

# Set the base directory (TEST/) based on the script's location
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# *** NEW: Define the Image Directory Path ***
IMG_DIR = BASE_DIR / "img"

# Input files
INPUT_FILE = OUTPUT_DIR / "penalty_event_log.csv"
GPAA_SUMMARY_FILE = OUTPUT_DIR / "goalkeeper_gpaa_summary.csv"

# Model paths
XG_MODEL_PATH = OUTPUT_DIR / "expected_goals_model.pkl"
SHOT_TENDENCY_MODEL_PATH = OUTPUT_DIR / "shot_tendency_lean_binary_model.pkl"

# --- Streamlit Setup (Aesthetics) ---
st.set_page_config(
    page_title="The 12-Yard Matchup Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for beautiful football-themed design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700;900&display=swap');

    /* Global Styling - Football Pitch Theme */
    .stApp {
        background: linear-gradient(135deg, #0a4d2e 0%, #0d1b17 50%, #1a1a2e 100%);
        color: #FAFAFA;
        font-family: 'Roboto', sans-serif;
    }

    /* Animated Background Pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            repeating-linear-gradient(90deg, 
                transparent, 
                transparent 49px, 
                rgba(255, 255, 255, 0.02) 49px, 
                rgba(255, 255, 255, 0.02) 50px);
        pointer-events: none;
        z-index: 0;
    }

    /* Main Content Layer */
    .main > div {
        position: relative;
        z-index: 1;
    }

    /* Epic Title with Glow Effect */
    .big-font {
        font-family: 'Bebas Neue', cursive;
        font-size: 4.5em !important;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff87 0%, #60efff 50%, #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3em;
        margin-top: 0.5em;
        text-shadow: 0 0 30px rgba(0, 255, 135, 0.5);
        letter-spacing: 4px;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(0, 255, 135, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(96, 239, 255, 0.8)); }
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #60efff;
        font-weight: 300;
        letter-spacing: 2px;
        margin-bottom: 2em;
        text-transform: uppercase;
    }

    /* VS Badge Styling */
    .vs-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 50px auto 0;
        font-family: 'Bebas Neue', cursive;
        font-size: 2.5em;
        color: white;
        box-shadow: 0 8px 32px rgba(238, 90, 111, 0.6),
                    inset 0 2px 8px rgba(255, 255, 255, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Selection Boxes - Stadium Style */
    .player-card {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.9) 0%, rgba(36, 59, 85, 0.9) 100%);
        border: 2px solid rgba(0, 255, 135, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .player-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 255, 135, 0.6);
        box-shadow: 0 15px 50px rgba(0, 255, 135, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Prediction Boxes - Goal Net Style */
    .prediction-box-left, .prediction-box-right {
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 10px;
        color: white;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }

    .prediction-box-left::before, .prediction-box-right::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(255, 255, 255, 0.03) 10px,
            rgba(255, 255, 255, 0.03) 20px
        );
        animation: net-pattern 20s linear infinite;
    }

    @keyframes net-pattern {
        0% { transform: translate(0, 0); }
        100% { transform: translate(20px, 20px); }
    }

    .prediction-box-left {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.6),
                    inset 0 1px 20px rgba(255, 255, 255, 0.1);
    }

    .prediction-box-left:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 50px rgba(59, 130, 246, 0.8);
    }

    .prediction-box-right {
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        box-shadow: 0 10px 40px rgba(153, 27, 27, 0.6),
                    inset 0 1px 20px rgba(255, 255, 255, 0.1);
    }

    .prediction-box-right:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 50px rgba(239, 68, 68, 0.8);
    }

    .prediction-percent {
        font-family: 'Bebas Neue', cursive;
        font-size: 5em;
        font-weight: 900;
        line-height: 1.0;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }

    .prediction-label {
        font-size: 1.3em;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }

    /* Metric Cards - Score Board Style */
    div[data-testid="stMetricValue"] {
        font-size: 2.5em !important;
        font-weight: 900 !important;
        color: #00ff87 !important;
        font-family: 'Bebas Neue', cursive !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.8) 0%, rgba(36, 59, 85, 0.8) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid rgba(0, 255, 135, 0.2);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 255, 135, 0.5);
        box-shadow: 0 8px 30px rgba(0, 255, 135, 0.2);
    }

    /* Button Styling - Stadium Lights */
    .stButton > button {
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        color: #0a1e2e;
        font-size: 1.5em;
        font-weight: 900;
        font-family: 'Bebas Neue', cursive;
        letter-spacing: 2px;
        padding: 20px 60px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 255, 135, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #60efff 0%, #00ff87 100%);
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 255, 135, 0.6);
    }

    /* Select Box Styling - REMOVED - Using Streamlit Default */

    /* Only keep dropdown menu styling for visibility */
    [role="listbox"] {
        background-color: #1C1E26 !important;
    }

    [role="option"] {
        background-color: #1C1E26 !important;
        color: #FFFFFF !important;
        padding: 10px !important;
    }

    [role="option"]:hover {
        background-color: rgba(0, 255, 135, 0.2) !important;
    }

    /* Headers with Underline Effect */
    h1, h2, h3 {
        font-family: 'Bebas Neue', cursive;
        letter-spacing: 2px;
        color: #00ff87;
        position: relative;
        padding-bottom: 10px;
    }

    h1::after, h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #00ff87 0%, transparent 100%);
        border-radius: 2px;
    }

    /* Container Borders - Pitch Lines */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border: 2px solid rgba(0, 255, 135, 0.2);
        border-radius: 15px;
        padding: 20px;
        background: rgba(20, 30, 48, 0.4);
        backdrop-filter: blur(10px);
    }

    /* Info/Warning Boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(96, 239, 255, 0.1) 0%, rgba(0, 255, 135, 0.1) 100%);
        border-left: 4px solid #60efff;
        border-radius: 10px;
        color: #FAFAFA;
    }

    /* Divider Styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #00ff87 50%, transparent 100%);
        margin: 40px 0;
    }

    /* Subheader Styling */
    .stSubheader {
        color: #60efff !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    /* Recommendation Box */
    .recommendation-box {
        padding: 30px;
        border: 3px solid #00ff87;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.1) 0%, rgba(96, 239, 255, 0.1) 100%);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0, 255, 135, 0.2);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Core Functions (Cached for Performance) ---

@st.cache_data
def calculate_elo_rating(df: pd.DataFrame, default_elo=1500, k_factor=32) -> pd.DataFrame:
    """Calculates dynamic Elo ratings for strikers and keepers."""

    # IMPORTANT: Assumes the goalkeeper column has already been renamed to 'goalkeeper_name'
    KEEPER_COLUMN = 'goalkeeper_name'

    if KEEPER_COLUMN not in df.columns:
        # Fallback error handling if rename failed (shouldn't happen with the new logic)
        st.error(f"Internal Error: Missing required column '{KEEPER_COLUMN}' in ELO calculation data.")
        return df

    df = df.sort_values(by=['match_date', 'minute', 'second']).reset_index(drop=True)
    striker_elos = {}
    keeper_elos = {}
    striker_elo_results = []

    df['outcome'] = (df['penalty_outcome'] == 'Goal').astype(float)

    for _, row in df.iterrows():
        striker = row['striker']
        keeper = row[KEEPER_COLUMN]

        striker_elo = striker_elos.get(striker, default_elo)
        striker_elo_results.append(striker_elo)
        keeper_elo = keeper_elos.get(keeper, default_elo)

        E_striker = 1 / (1 + 10 ** ((keeper_elo - striker_elo) / 400))

        striker_new_elo = striker_elo + k_factor * (row['outcome'] - E_striker)
        keeper_new_elo = keeper_elo + k_factor * ((1 - row['outcome']) - (1 - E_striker))

        striker_elos[striker] = striker_new_elo
        keeper_elos[keeper] = keeper_new_elo

    elo_df = pd.DataFrame({'striker_elo_prev': striker_elo_results})
    return df.join(elo_df[['striker_elo_prev']])


def rename_goalkeeper_column(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Checks for common variations of the goalkeeper column name and renames it
    to 'goalkeeper_name' for consistency.
    """
    # Print columns for mandatory user debugging if auto-rename fails
    print("-" * 50)
    print(f"DEBUG: Columns in {df_name}:")
    print(df.columns.tolist())
    print("-" * 50)

    # Common variations of the goalkeeper column name
    common_keeper_cols = ['goalkeeper_name', 'goalkeeper', 'Goalkeeper', 'keeper', 'Keeper', 'Goalkeeper Name',
                          'Keeper Name']

    found_col = None
    for col in common_keeper_cols:
        if col in df.columns:
            found_col = col
            break

    if found_col and found_col != 'goalkeeper_name':
        # Rename the found column to the expected standard name
        print(f"INFO: Renaming column '{found_col}' in {df_name} to 'goalkeeper_name'.")
        df = df.rename(columns={found_col: 'goalkeeper_name'})

    elif 'goalkeeper_name' not in df.columns:
        # If the column isn't found and isn't the expected name, this is the root cause.
        # This will still trigger a KeyError downstream if the data relies on it.
        print(f"ERROR: Could not find any common goalkeeper column name in {df_name}.")
        st.error(
            f"Data Error: Could not find the goalkeeper column in {df_name}. Please check the console output to see the correct column names and manually update the code.")

    return df


@st.cache_resource
def load_models_and_data():
    """Loads all models and data, running ELO calculation only once."""
    try:
        raw_df = pd.read_csv(INPUT_FILE)
        gpaa_df = pd.read_csv(GPAA_SUMMARY_FILE)

        # --- FIX: Standardize the Goalkeeper Name Column in both DataFrames ---
        raw_df = rename_goalkeeper_column(raw_df, "penalty_event_log.csv")
        gpaa_df = rename_goalkeeper_column(gpaa_df, "goalkeeper_gpaa_summary.csv")

        # Run the intensive ELO calculation once (requires 'goalkeeper_name' in raw_df)
        df_engineered = calculate_elo_rating(raw_df.copy())

        # Load the models
        xg_model = joblib.load(XG_MODEL_PATH)
        shot_tendency_model = joblib.load(SHOT_TENDENCY_MODEL_PATH)

        # Generate unique player lists for dropdowns (requires 'goalkeeper_name' in df_engineered)
        strikers = sorted(df_engineered['striker'].unique().tolist())
        keepers = sorted(df_engineered['goalkeeper_name'].unique().tolist())

        return df_engineered, gpaa_df, xg_model, shot_tendency_model, strikers, keepers

    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure all model scripts were run. Missing: {e.filename}")
        st.stop()
    except Exception as e:
        # Catch and display the error, but the debug print should reveal the column name
        st.error(f"Error loading models or data: {e}. Check your terminal for column name debugging information.")
        st.stop()


# --- Prediction and Report Generation Logic (unchanged) ---

def predict_xg_and_lean(df_engineered, xg_model, shot_tendency_model, striker_name):
    """Calculates xG probability and the binary shot tendency prediction."""

    # 1. Calculate xG
    xg_features = ['shot_angle', 'shot_height', 'match_scoreline_diff', 'footedness', 'home_or_away', 'shot_side']

    df_predict = df_engineered[df_engineered['shot_angle'].notna() & df_engineered['shot_height'].notna()].copy()
    df_engineered['xg_probability'] = np.nan

    if not df_predict.empty and all(col in df_predict.columns for col in xg_features):
        try:
            X_xg = df_predict[xg_features].fillna(0)
            df_predict['xg_probability'] = xg_model.predict_proba(X_xg)[:, 1]
            df_engineered.update(df_predict[['xg_probability']])
        except Exception as e:
            pass

    # Filter for the specific striker's historical penalties
    striker_data = df_engineered[df_engineered['striker'] == striker_name].copy()

    # Calculate key metrics
    latest_elo = striker_data['striker_elo_prev'].iloc[-1].round(0) if not striker_data.empty else 1500.0
    avg_xg_series = striker_data[striker_data['shot_angle'].notna()]['xg_probability'].mean()
    avg_xg = avg_xg_series.round(3) if pd.notna(avg_xg_series) else 0.000

    # 2. Predict Actionable Lean

    required_features = [
        'keeper_elo_prev', 'minute', 'match_scoreline_diff',
        'home_or_away', 'hist_left_pct', 'cum_total_prev',
        'elo_advantage', 'footedness'
    ]
    target_mapping = ['Left', 'Right']

    predicted_probs = [0.5, 0.5]  # Default 50/50 fallback

    if not striker_data.empty:
        last_penalty = striker_data.iloc[[-1]].copy()

        last_penalty['keeper_elo_prev'] = 1500.0  # Mock average keeper ELO
        last_penalty['elo_advantage'] = last_penalty['striker_elo_prev'] - last_penalty['keeper_elo_prev']
        last_penalty['cum_total_prev'] = len(striker_data) - 1

        left_shots_count = striker_data['shot_side'].value_counts().get('Left', 0)
        total_shots = len(striker_data)
        last_penalty['hist_left_pct'] = (left_shots_count / total_shots) if total_shots > 0 else 0.0

        for col in required_features:
            if col not in last_penalty.columns:
                last_penalty[col] = 0.0 if col in ['minute', 'match_scoreline_diff'] else 'Unknown'

        mock_input = last_penalty[required_features].fillna(0).head(1)

        try:
            predicted_probs = shot_tendency_model.predict_proba(mock_input)[0]
        except Exception as e:
            pass

    prediction_df = pd.DataFrame({
        'Direction': target_mapping,
        'Probability': predicted_probs
    }).sort_values(by='Probability', ascending=False)

    most_likely_side = prediction_df['Direction'].iloc[0]

    # Store results in a dictionary
    report_data = {
        'striker_data': striker_data,
        'latest_elo': latest_elo,
        'avg_xg': avg_xg,
        'most_likely_side': most_likely_side,
        'prediction_df': prediction_df
    }
    return report_data


# --- Graphing Functions (unchanged) ---

def plot_striker_shot_map(striker_data: pd.DataFrame):
    """Generates a scatter plot of historical shot locations."""

    # Use dummy coordinates for visual separation of L, C, R
    plot_data = striker_data[striker_data['shot_side'].isin(['Left', 'Center', 'Right'])].copy()
    if plot_data.empty:
        return st.info("No shot location data available for plotting.")

    # Assign Y position for visual separation
    def assign_y_position(side):
        if side == 'Left': return 2
        if side == 'Center': return 1
        return 0

    plot_data['y_pos'] = plot_data['shot_side'].apply(assign_y_position)

    # Determine color by outcome
    plot_data['color'] = plot_data['penalty_outcome'].apply(lambda x: 'Goal' if x == 'Goal' else 'Miss/Save')

    chart = alt.Chart(plot_data).mark_circle(size=150).encode(
        x=alt.X('shot_side', title="Historical Shot Side", sort=['Left', 'Center', 'Right']),
        y=alt.Y('y_pos', axis=None),  # Hide y-axis as it's arbitrary
        color=alt.Color('color', scale=alt.Scale(domain=['Goal', 'Miss/Save'], range=['#00ff87', '#ff6b6b']),
                        legend=alt.Legend(title="Outcome")),
        tooltip=['match_date', 'penalty_outcome', 'shot_side']
    ).properties(
        title='Historical Shot Distribution (Scatter Plot)'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False,
        labelColor='#FAFAFA',
        titleColor='#00ff87'
    ).configure_title(
        color='#00ff87',
        fontSize=16
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def plot_elo_trend(striker_data: pd.DataFrame):
    """Generates a line chart of the striker's ELO rating over time."""
    if striker_data.empty or 'striker_elo_prev' not in striker_data.columns:
        return st.info("No ELO trend data available for plotting.")

    elo_df = striker_data[['striker_elo_prev']].reset_index().rename(columns={'index': 'Penalty Number'})

    chart = alt.Chart(elo_df).mark_line(point=True, color='#60efff', strokeWidth=3).encode(
        x=alt.X('Penalty Number', axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('striker_elo_prev', title="ELO Rating"),
        tooltip=['Penalty Number', 'striker_elo_prev']
    ).properties(
        title='Striker ELO Rating Trend Over Career'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False,
        labelColor='#FAFAFA',
        titleColor='#00ff87'
    ).configure_title(
        color='#00ff87',
        fontSize=16
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def plot_keeper_save_distribution(df: pd.DataFrame, keeper_name: str):
    """Generates a bar chart of the selected keeper's saves by side."""
    keeper_data = df[df['goalkeeper_name'] == keeper_name].copy()
    if keeper_data.empty:
        return st.info("No penalty events found for this goalkeeper.")

    save_data = keeper_data[keeper_data['penalty_outcome'] == 'Saved'].groupby('shot_side').size().reset_index(
        name='Saves')

    if save_data.empty:
        return st.info(f"The selected keeper ({keeper_name}) has no recorded saves in the dataset.")

    chart = alt.Chart(save_data).mark_bar(color='#00ff87').encode(
        x=alt.X('shot_side', title="Shot Side Saved", sort=['Left', 'Center', 'Right']),
        y=alt.Y('Saves', title="Number of Saves"),
        tooltip=['shot_side', 'Saves']
    ).properties(
        title=f'{keeper_name} - Saves by Shot Direction'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False,
        labelColor='#FAFAFA',
        titleColor='#00ff87'
    ).configure_title(
        color='#00ff87',
        fontSize=16
    )

    st.altair_chart(chart, use_container_width=True)


# --- Main Streamlit App Layout ---

def main():
    st.markdown('<p class="big-font">⚽ THE 12-YARD MATCHUP PREDICTOR ⚽</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Penalty Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load all data and models (cached)
    df_engineered, gpaa_df, xg_model, shot_tendency_model, strikers, keepers = load_models_and_data()

    # Set default values for initial run
    default_striker = strikers[0] if strikers else None
    default_keeper = keepers[0] if keepers else None

    # Find indices for default selections
    striker_idx = 0
    keeper_idx = 0

    # 1. Selection Layout
    col1, col_vs, col2 = st.columns([4, 1, 4])

    with col1:
        st.subheader("⚡ Striker")
        selected_striker = st.selectbox(
            "Select Striker:",
            options=strikers,
            index=striker_idx,
            key="striker_select",
            label_visibility="collapsed",
            placeholder="Choose a striker..."
        )

    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🧤 Goalkeeper")
        selected_keeper = st.selectbox(
            "Select Goalkeeper:",
            options=keepers,
            index=keeper_idx,
            key="keeper_select",
            label_visibility="collapsed",
            placeholder="Choose a goalkeeper..."
        )

    st.markdown("---")

    # 2. Analysis Trigger
    if st.button("⚡ ANALYZE DUEL", use_container_width=True, type="primary"):

        # Run prediction logic
        report_data = predict_xg_and_lean(df_engineered, xg_model, shot_tendency_model, selected_striker)
        striker_data = report_data['striker_data']

        if striker_data.empty:
            st.error(f"No penalty data found for striker: '{selected_striker}'")
            return

        # Get keeper GPAA data
        filtered_gpaa = gpaa_df[gpaa_df['goalkeeper_name'] == selected_keeper]
        keeper_gpaa = filtered_gpaa.iloc[0] if not filtered_gpaa.empty else None

        # --- Section A: Head-to-Head & Tactical Recommendation ---

        st.header(f"🎯 Tactical Matchup: {selected_striker} vs. {selected_keeper}")

        # H2H History Check
        h2h_data = striker_data[striker_data['goalkeeper_name'] == selected_keeper]

        with st.container(border=True):
            st.subheader("📊 Head-to-Head History")
            if not h2h_data.empty:
                penalties_faced = len(h2h_data)
                goals_scored = (h2h_data['penalty_outcome'] == 'Goal').sum()
                saves_made = (h2h_data['penalty_outcome'] == 'Saved').sum()

                col_h2h1, col_h2h2, col_h2h3 = st.columns(3)
                col_h2h1.metric("Penalties Faced", penalties_faced)
                col_h2h2.metric("Goals Scored", goals_scored, delta_color="inverse")
                col_h2h3.metric("Saves Made", saves_made, delta_color="normal")
            else:
                st.info(f"No recorded head-to-head history found between {selected_striker} and {selected_keeper}.")

        st.subheader("🎯 Actionable Prediction: Where is the Striker Leaning?")

        # Actionable Lean Visualization
        left_prob = (report_data['prediction_df'][report_data['prediction_df']['Direction'] == 'Left'][
                         'Probability'].iloc[0] * 100).round(1)
        right_prob = (report_data['prediction_df'][report_data['prediction_df']['Direction'] == 'Right'][
                          'Probability'].iloc[0] * 100).round(1)

        col_pred_L, col_pred_R = st.columns(2)

        with col_pred_L:
            st.markdown(f"""
            <div class="prediction-box-left">
                <div class="prediction-label">◀ LEFT SIDE</div>
                <div class="prediction-percent">{left_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_pred_R:
            st.markdown(f"""
            <div class="prediction-box-right">
                <div class="prediction-label">RIGHT SIDE ▶</div>
                <div class="prediction-percent">{right_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("💡 Final Tactical Recommendation")

        # Final Recommendation Summary
        recommendation_side = report_data['prediction_df'].iloc[0]['Direction'].upper()
        recommendation_prob = (report_data['prediction_df'].iloc[0]['Probability'] * 100).round(1)

        summary = f"""
        <div class="recommendation-box">
            <h3 style="color: #00ff87; margin-top: 0;">🎯 STRATEGIC INSIGHT</h3>
            The model predicts that when <strong>{selected_striker}</strong> avoids the center, he has a <strong style="color: #00ff87;">{recommendation_prob}%</strong> lean toward the <strong style="color: #60efff;">{recommendation_side}</strong> side. 
            <br><br>
            <strong style="color: #ff6b6b;">⚡ Keeper Advice:</strong> The optimal strategy for <strong>{selected_keeper}</strong> is to commit to the <strong style="color: #60efff;">{recommendation_side}</strong>. {selected_striker}'s high ELO (<strong style="color: #00ff87;">{report_data['latest_elo']:.0f}</strong>) and strong average xG (<strong style="color: #00ff87;">{report_data['avg_xg']}</strong>) mean any deviation from the predicted path is likely to result in a goal.
        </div>
        """
        st.markdown(summary, unsafe_allow_html=True)

        st.markdown("---")

        # --- Section B: Striker Analysis (Metrics and Graphs) ---

        st.header("⚡ Detailed Striker Profile")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Latest ELO Rating", f"{report_data['latest_elo']:.0f}")
        col_m2.metric("Average Shot Quality (xG)", f"{report_data['avg_xg']}",
                      help="Expected Goals per shot, measures placement quality.")
        col_m3.metric("Total Penalties", len(striker_data))

        # Striker Graphs
        st.subheader("📈 Striker Visualization")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            plot_striker_shot_map(striker_data)
        with col_g2:
            plot_elo_trend(striker_data)

        st.markdown("---")

        # --- Section C: Goalkeeper Analysis (Metrics and Graphs) ---

        st.header("🧤 Detailed Goalkeeper Profile")

        if keeper_gpaa is not None:
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Goalkeeper GPAA", f"{keeper_gpaa['gpaa']:.2f}",
                          help="Goals Prevented Above Average: Positive is good.")
            col_k2.metric("xG Faced", f"{keeper_gpaa['total_expected_goals']:.2f}")
            col_k3.metric("Goals Conceded", f"{keeper_gpaa['actual_goals_conceded']:.0f}")
        else:
            st.warning(f"Keeper data not found for {selected_keeper}.")

        # Keeper Graphs
        col_kg1, col_kg2 = st.columns(2)
        with col_kg1:
            plot_keeper_save_distribution(df_engineered, selected_keeper)

        with col_kg2:
            st.subheader("🏆 Top GPAA Keepers (Rank)")
            top_keepers_plot = gpaa_df.head(10).copy()
            top_keepers_plot['color'] = np.where(top_keepers_plot['goalkeeper_name'] == selected_keeper,
                                                 'Selected Keeper', 'Other')

            chart = alt.Chart(top_keepers_plot).mark_bar().encode(
                x=alt.X('goalkeeper_name', sort='-y', title="Goalkeeper"),
                y=alt.Y('gpaa', title="GPAA Score"),
                color=alt.Color('color',
                                scale=alt.Scale(domain=['Selected Keeper', 'Other'], range=['#ff6b6b', '#60efff']),
                                legend=None),
                tooltip=['goalkeeper_name', 'gpaa', 'penalties_faced']
            ).properties(
                title='Top 10 Keepers by GPAA'
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=False,
                labelColor='#FAFAFA',
                titleColor='#00ff87'
            ).configure_title(
                color='#00ff87',
                fontSize=16
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # --- Section D: Data Attribution ---
        st.header("📚 Methodology & Data Provenance")
        st.markdown("""
        <div style="background: rgba(20, 30, 48, 0.6); padding: 20px; border-radius: 15px; border: 2px solid rgba(0, 255, 135, 0.2);">
        This analysis is powered by statistical modeling applied to the <strong style="color: #00ff87;">StatsBomb Open Data</strong> penalty event logs. 
        <br><br>
        The data was cleaned, processed from the original JSON format, and feature-engineered to calculate advanced metrics 
        like ELO Ratings, Expected Goals (xG), and Goals Prevented Above Average (GPAA). The 'Actionable Prediction' 
        uses a machine learning model trained on match situation and historical trends to predict the most likely shot direction.
        </div>
        """, unsafe_allow_html=True)

        # StatsBomb Logo
        # *** MODIFIED LINE: Use the path defined using Pathlib for reliability ***
        logo_path = str(IMG_DIR / "logo.png")
        st.image(logo_path, caption="Data provided by StatsBomb Open Data", width=150)


if __name__ == "__main__":
    main()
