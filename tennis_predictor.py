import pandas as pd
from datetime import date
import joblib
import os

# Define constants - ensure these match your EDA notebook
ROLLING_WINDOW = 10
ROLLING_DERIVED_COLS = [
    '1st_serve_in_pct', '1st_serve_win_pct', '2nd_serve_win_pct',
    'break_point_save_pct', 'total_pts_won_on_serve_pct', 'break_pct'
]

# Corrected Path Definitions:

# Get the directory of the current script (tennis_predictor.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level from the script's directory to reach the project root 'TennisForcaster'
# Assuming tennis_predictor.py is directly under TennisForcaster/
PROJECT_ROOT = current_script_dir

# Now define your data and model directories relative to PROJECT_ROOT
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

PLAYER_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'player_df_historical.joblib')
MODEL_PATH = os.path.join(MODEL_DIR, 'tennis_match_predictor.joblib') # Corrected model filename

# Load the historical player data once
try:
    player_df_historical = joblib.load(PLAYER_DF_PATH)
    print(f"Loaded player historical data from {PLAYER_DF_PATH}")
except FileNotFoundError:
    print(f"Error: player_df_historical.joblib not found at {PLAYER_DF_PATH}. Please run EDA notebook first.")
    player_df_historical = pd.DataFrame() # Create empty to avoid immediate errors

# Load the trained model once
try:
    trained_model = joblib.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")
    # Store feature names for consistent ordering during prediction
    # This is CRUCIAL: features MUST be in the same order as training
    MODEL_FEATURES_ORDER = trained_model.feature_names_in_
except FileNotFoundError:
    print(f"Error: trained_model.joblib not found at {MODEL_PATH}. Please train model in EDA notebook first.")
    trained_model = None # Set to None to handle later
    MODEL_FEATURES_ORDER = [] # Empty list to avoid errors


def get_player_stats_for_match(player_id, opponent_id, tourney_date, player_df_hist):
    """
    Extracts a player's pre-match historical stats and their H2H stats against a specific opponent.
    player_df_hist should be sorted by player_id and tourney_date.
    """
    player_stats = player_df_hist[
        (player_df_hist['player_id'] == player_id) &
        (player_df_hist['tourney_date'] < tourney_date) # Crucial: stats from *before* the current match
    ].sort_values(by='tourney_date', ascending=False)

    # Get last available stats before the match for 'prev' features
    last_match_stats = player_stats.iloc[0] if not player_stats.empty else {}

    # Get H2H stats against the specific opponent
    h2h_stats = player_df_hist[
        (player_df_hist['player_id'] == player_id) &
        (player_df_hist['opponent_id'] == opponent_id) & # Specific opponent
        (player_df_hist['tourney_date'] < tourney_date)
    ]
    # Sum/count all H2H matches before the current date
    h2h_wins = h2h_stats['is_winner'].sum() if not h2h_stats.empty else 0
    h2h_matches = h2h_stats.shape[0] if not h2h_stats.empty else 0
    h2h_win_pct = h2h_wins / h2h_matches if h2h_matches > 0 else 0.5 # Default to 0.5 for no H2H

    # Prepare a dictionary of extracted stats
    extracted_stats = {
        'prev_rank': last_match_stats.get('player_prev_rank'), # These should already be shifted in player_df_historical
        'prev_rank_points': last_match_stats.get('player_prev_rank_points'),
        'prev_age': last_match_stats.get('player_prev_age'),
        'prev_ht': last_match_stats.get('player_prev_ht'),
        f'avg_is_winner_last{ROLLING_WINDOW}': last_match_stats.get(f'player_avg_is_winner_last{ROLLING_WINDOW}'),
        f'avg_is_winner_on_surface_last{ROLLING_WINDOW}': last_match_stats.get(f'player_avg_is_winner_on_surface_last{ROLLING_WINDOW}'),
        'h2h_wins_vs_opponent': h2h_wins,
        'h2h_matches_vs_opponent': h2h_matches,
        'h2h_win_pct_vs_opponent': h2h_win_pct
    }

    # Add other rolling derived columns
    for col_base in ROLLING_DERIVED_COLS:
        extracted_stats[f'avg_{col_base}_last{ROLLING_WINDOW}'] = last_match_stats.get(f'player_avg_{col_base}_last{ROLLING_WINDOW}')
        extracted_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}'] = last_match_stats.get(f'player_avg_{col_base}_on_surface_last{ROLLING_WINDOW}')

    # Handle NaNs from missing history (e.g., a new player, or first few matches)
    # This must match your EDA's imputation strategy
    for key, value in extracted_stats.items():
        if pd.isna(value):
            if 'win_pct' in key or 'is_winner' in key:
                extracted_stats[key] = 0.5 # Default win rates to 0.5 for unknown
            elif 'rank' in key:
                extracted_stats[key] = 9999 # Default rank to a high value
            elif 'ht' in key or 'age' in key:
                # Use a general average from your training data, or 0/mean if not available
                # For simplicity, let's use a placeholder. In production, save training means.
                extracted_stats[key] = 0 # Or some global average
            elif 'h2h_matches' in key:
                extracted_stats[key] = 0
            else:
                extracted_stats[key] = 0 # Default other stats to 0 or mean of training data

    return extracted_stats


def prepare_single_match_features(player1_id, player2_id, tourney_date, player_df_hist):
    """
    Prepares a single row of features for prediction for a match between player1 and player2.
    """
    p1_stats = get_player_stats_for_match(player1_id, player2_id, tourney_date, player_df_hist)
    p2_stats = get_player_stats_for_match(player2_id, player1_id, tourney_date, player_df_hist) # Note: P2's opponent is P1!

    features = {}

    # Static Differences
    features['rank_diff'] = p1_stats['prev_rank'] - p2_stats['prev_rank']
    features['rank_points_diff'] = p1_stats['prev_rank_points'] - p2_stats['prev_rank_points']
    features['age_diff'] = p1_stats['prev_age'] - p2_stats['prev_age']
    features['height_diff'] = p1_stats['prev_ht'] - p2_stats['prev_ht']

    # General Rolling Average Differences
    features[f'avg_win_pct_diff_last{ROLLING_WINDOW}'] = \
        p1_stats[f'avg_is_winner_last{ROLLING_WINDOW}'] - \
        p2_stats[f'avg_is_winner_last{ROLLING_WINDOW}']

    for col_base in ROLLING_DERIVED_COLS:
        features[f'avg_{col_base}_diff'] = \
            p1_stats[f'avg_{col_base}_last{ROLLING_WINDOW}'] - \
            p2_stats[f'avg_{col_base}_last{ROLLING_WINDOW}']

    # Surface-Specific Rolling Average Differences
    features[f'avg_win_pct_on_surface_diff_last{ROLLING_WINDOW}'] = \
        p1_stats[f'avg_is_winner_on_surface_last{ROLLING_WINDOW}'] - \
        p2_stats[f'avg_is_winner_on_surface_last{ROLLING_WINDOW}']

    for col_base in ROLLING_DERIVED_COLS:
        features[f'avg_{col_base}_on_surface_diff_last{ROLLING_WINDOW}'] = \
            p1_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}'] - \
            p2_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}']

    # Head-to-Head (H2H) Features
    features['h2h_win_pct_diff'] = \
        p1_stats['h2h_win_pct_vs_opponent'] - \
        p2_stats['h2h_win_pct_vs_opponent']

    features['h2h_matches_total'] = p1_stats['h2h_matches_vs_opponent'] + p2_stats['h2h_matches_vs_opponent']


    # Convert to DataFrame to ensure consistent column order
    features_df = pd.DataFrame([features])

     # Reorder columns to match the training data
    # This is absolutely vital for the model to work correctly!
    # Corrected line: Use len() to check if the list/array is empty
    if len(MODEL_FEATURES_ORDER) > 0: # <-- CHANGE THIS LINE
        features_df = features_df[MODEL_FEATURES_ORDER]
    else:
        print("Warning: MODEL_FEATURES_ORDER not available. Prediction may fail due to incorrect feature order.")
        # Fallback for development/debugging, in production you'd raise an error

    return features_df


def predict_match_outcome(player1_id, player2_id, tourney_date):
    """
    Predicts the outcome of a match between two players.
    Returns 1 if Player 1 is predicted to win, 0 if Player 2 is predicted to win.
    tourney_date can be a datetime.date object.
    """
    if trained_model is None:
        print("Model not loaded. Cannot make prediction.")
        return None

    # Convert tourney_date to pandas Timestamp for consistent comparison
    # This is the CRUCIAL CHANGE
    tourney_date_ts = pd.Timestamp(tourney_date)

    # Prepare features for the match, passing the Timestamp object
    match_features = prepare_single_match_features(player1_id, player2_id, tourney_date_ts, player_df_historical)

    if match_features.empty:
        print("Could not prepare features for the match. Prediction aborted.")
        return None

    # Make prediction
    prediction_proba = trained_model.predict_proba(match_features)[0] # Get probabilities for [0, 1]
    prediction = trained_model.predict(match_features)[0] # Get class prediction

    print(f"Prediction for P1 (ID: {player1_id}) vs P2 (ID: {player2_id}) on {tourney_date}:")
    print(f"Probability P1 wins: {prediction_proba[1]:.4f}")
    print(f"Probability P2 wins: {prediction_proba[0]:.4f}") # P2 wins means P1 loses, so probability for label 0
    print(f"Predicted Winner (1=P1, 0=P2): {prediction}")

    return prediction, prediction_proba

# Example Usage (for testing tennis_predictor.py directly)
if __name__ == '__main__':
    # You'll need actual player IDs and a tournament date from your dataset
    # to test this properly. Example from Wimbledon 2023 final:
    # Carlos Alcaraz (ID: 200780) vs Novak Djokovic (ID: 104925)
    # tourney_date should be a pandas Timestamp or datetime.date object
    from datetime import date
    alcaraz_id = 207989
    djokovic_id = 104925
    wimbledon_final_date = date(2023, 7, 16) # This remains a date object

    print("\n--- Testing Prediction Function ---")
    if not player_df_historical.empty and trained_model is not None:
        # No change needed here for passing wimbledon_final_date,
        # as the conversion happens inside predict_match_outcome
        predicted_outcome, probabilities = predict_match_outcome(alcaraz_id, djokovic_id, wimbledon_final_date)
        if predicted_outcome is not None:
            print(f"Final Prediction: Player {alcaraz_id if predicted_outcome == 1 else djokovic_id} is predicted to win.")
    else:
        print("Cannot run example prediction: data or model not loaded.")
