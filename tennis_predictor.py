import pandas as pd
from datetime import date
import joblib
import os
import numpy as np

# TensorFlow and Keras imports are necessary to load the model
import tensorflow as tf
from tensorflow import keras

# Define constants - ensure these match your EDA notebook
ROLLING_WINDOW = 10
ROLLING_DERIVED_COLS = [
    'is_winner', '1st_serve_in_pct', '1st_serve_win_pct', '2nd_serve_win_pct',
    'break_point_save_pct', 'total_pts_won_on_serve_pct', 'break_pct'
]

# Corrected Path Definitions:
current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = current_script_dir
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

PLAYER_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'player_df_historical.joblib')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'tennis_match_predictor_nn_fixed.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_nn_fixed.joblib')

# Load the historical player data once
try:
    player_df_historical = joblib.load(PLAYER_DF_PATH)
    print(f"Loaded player historical data from {PLAYER_DF_PATH}")
except FileNotFoundError:
    print(f"Error: player_df_historical.joblib not found at {PLAYER_DF_PATH}. Please run EDA notebook first.")
    player_df_historical = pd.DataFrame()

# Load the trained Keras model and the StandardScaler
trained_model = None
scaler = None

try:
    trained_model = keras.models.load_model(KERAS_MODEL_PATH)
    print(f"Loaded trained Keras model from {KERAS_MODEL_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded StandardScaler from {SCALER_PATH}")
except (FileNotFoundError, ValueError) as e:
    print(f"Error: Required model or data file not found. Details: {e}")
    trained_model = None
    scaler = None

def get_player_stats_for_match(player_id, tourney_date, player_df_hist):
    """
    Extracts a player's pre-match historical stats.
    """
    player_stats = player_df_hist[
        (player_df_hist['player_id'] == player_id) &
        (player_df_hist['tourney_date'] < tourney_date)
    ].sort_values(by='tourney_date', ascending=False)
    
    last_match_stats = player_stats.iloc[0] if not player_stats.empty else pd.Series()
    
    extracted_stats = {
        'prev_rank': last_match_stats.get('player_prev_rank'),
        'prev_rank_points': last_match_stats.get('player_prev_rank_points'),
        'prev_age': last_match_stats.get('player_prev_age'),
        'prev_ht': last_match_stats.get('player_prev_ht'),
    }

    for col_base in ROLLING_DERIVED_COLS:
        extracted_stats[f'avg_{col_base}_last{ROLLING_WINDOW}'] = last_match_stats.get(f'player_avg_{col_base}_last{ROLLING_WINDOW}')
        extracted_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}'] = last_match_stats.get(f'player_avg_{col_base}_on_surface_last{ROLLING_WINDOW}')

    # Fill missing values with defaults
    for key, value in extracted_stats.items():
        if pd.isna(value):
            if 'win_pct' in key or 'is_winner' in key:
                extracted_stats[key] = 0.5
            elif 'rank' in key:
                extracted_stats[key] = 9999
            elif 'ht' in key or 'age' in key:
                extracted_stats[key] = 0
            else:
                extracted_stats[key] = 0
    return extracted_stats

def prepare_single_match_features_original_format(player1_id, player2_id, tourney_date, surface='Hard', best_of=3):
    """
    Prepares features in the original format that matches the training data.
    This assumes your training data used individual player stats, not differences.
    """
    # Get player stats for both players
    p1_stats = get_player_stats_for_match(player1_id, tourney_date, player_df_historical)
    p2_stats = get_player_stats_for_match(player2_id, tourney_date, player_df_historical)
    
    # Create a row that matches the training format
    # This assumes your training data had features for both players and match conditions
    features = {}
    
    # Player 1 features (assuming these were prefixed or in a specific order)
    features['player_prev_rank'] = p1_stats['prev_rank']
    features['player_prev_rank_points'] = p1_stats['prev_rank_points'] 
    features['player_prev_age'] = p1_stats['prev_age']
    features['player_prev_ht'] = p1_stats['prev_ht']
    
    # Player 1 rolling averages
    for col_base in ROLLING_DERIVED_COLS:
        features[f'player_avg_{col_base}_last{ROLLING_WINDOW}'] = p1_stats[f'avg_{col_base}_last{ROLLING_WINDOW}']
        features[f'player_avg_{col_base}_on_surface_last{ROLLING_WINDOW}'] = p1_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}']
    
    # Opponent features
    features['opponent_prev_rank'] = p2_stats['prev_rank']
    features['opponent_prev_rank_points'] = p2_stats['prev_rank_points']
    features['opponent_prev_age'] = p2_stats['prev_age'] 
    features['opponent_prev_ht'] = p2_stats['prev_ht']
    
    # Opponent rolling averages
    for col_base in ROLLING_DERIVED_COLS:
        features[f'opponent_avg_{col_base}_last{ROLLING_WINDOW}'] = p2_stats[f'avg_{col_base}_last{ROLLING_WINDOW}']
        features[f'opponent_avg_{col_base}_on_surface_last{ROLLING_WINDOW}'] = p2_stats[f'avg_{col_base}_on_surface_last{ROLLING_WINDOW}']
    
    # Match conditions (these appeared in the error message)
    features['best_of'] = best_of
    
    # Surface encoding (you'll need to check how this was done in training)
    surface_mapping = {'Hard': 0, 'Clay': 1, 'Grass': 2}  # Adjust based on your encoding
    features['surface'] = surface_mapping.get(surface, 0)
    
    return pd.DataFrame([features])

def get_expected_feature_names():
    """
    Try to get the expected feature names from the scaler.
    """
    if scaler is None:
        return []
    
    if hasattr(scaler, 'feature_names_in_'):
        return list(scaler.feature_names_in_)
    else:
        print("Scaler doesn't have feature_names_in_ attribute")
        return []

def prepare_features_matching_training(player1_id, player2_id, tourney_date, surface='Hard', best_of=3):
    """
    Prepare features to exactly match what was used during training.
    You'll need to adjust this based on your actual training feature set.
    """
    expected_features = get_expected_feature_names()
    
    if not expected_features:
        print("Cannot determine expected features from scaler")
        return None
    
    print(f"Expected features ({len(expected_features)}):")
    for i, feature in enumerate(expected_features):
        print(f"{i+1:2d}. {feature}")
    
    # Get player stats
    p1_stats = get_player_stats_for_match(player1_id, tourney_date, player_df_historical)
    p2_stats = get_player_stats_for_match(player2_id, tourney_date, player_df_historical)
    
    # Build feature dictionary matching expected names
    features = {}
    
    # This is where you need to map your player stats to the expected feature names
    # Based on the error message, some expected features include:
    # '1st_serve_in_pct', '1st_serve_win_pct', '2nd_serve_win_pct', 'best_of', 'break_pct', etc.
    
    for expected_feature in expected_features:
        if expected_feature == 'best_of':
            features[expected_feature] = best_of
        elif expected_feature == 'surface':
            surface_mapping = {'Hard': 0, 'Clay': 1, 'Grass': 2}
            features[expected_feature] = surface_mapping.get(surface, 0)
        elif expected_feature in ['1st_serve_in_pct', '1st_serve_win_pct', '2nd_serve_win_pct', 'break_pct', 
                                  'break_point_save_pct', 'total_pts_won_on_serve_pct']:
            # These might be player 1 stats directly
            mapped_key = f'avg_{expected_feature}_last{ROLLING_WINDOW}'
            features[expected_feature] = p1_stats.get(mapped_key, 0.5 if 'pct' in expected_feature else 0)
        # Add more mappings as needed based on the actual expected features
        else:
            # Default value for unrecognized features
            features[expected_feature] = 0
    
    return pd.DataFrame([features])

def predict_match_outcome(player1_id, player2_id, tourney_date, surface='Hard', best_of=3):
    """
    Predicts the outcome of a match between two players.
    """
    if trained_model is None or scaler is None:
        print("Model or scaler not loaded. Cannot make prediction.")
        return None, None

    tourney_date_ts = pd.Timestamp(tourney_date)
    
    # Try to prepare features matching the training format
    match_features_df = prepare_features_matching_training(player1_id, player2_id, tourney_date_ts, surface, best_of)
    
    if match_features_df is None or match_features_df.empty:
        print("Could not prepare features for the match. Prediction aborted.")
        return None, None
    
    print(f"\nPrepared features shape: {match_features_df.shape}")
    print("Feature values:")
    for col in match_features_df.columns:
        print(f"  {col}: {match_features_df[col].iloc[0]}")
    
    try:
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(match_features_df)
        
        # Make the prediction
        prediction_proba = trained_model.predict(scaled_features, verbose=0).ravel()[0]
        prediction = 1 if prediction_proba > 0.5 else 0

        print(f"\nPrediction for P1 (ID: {player1_id}) vs P2 (ID: {player2_id}) on {tourney_date}:")
        print(f"Surface: {surface}, Best of: {best_of}")
        print(f"Probability P1 wins: {prediction_proba:.4f}")
        print(f"Probability P2 wins: {1 - prediction_proba:.4f}")
        print(f"Predicted Winner (1=P1, 0=P2): {prediction}")

        return prediction, prediction_proba
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Example Usage
if __name__ == '__main__':
    alcaraz_id = 207989
    sinner_id = 206173
    us_open_final_date = date(2025, 9, 7)

    print("\n--- Testing Prediction Function ---")
    if not player_df_historical.empty and trained_model is not None and scaler is not None:
        predicted_outcome, probabilities = predict_match_outcome(
            alcaraz_id, sinner_id, us_open_final_date, surface='Hard', best_of=5
        )
        if predicted_outcome is not None:
            winner_id = alcaraz_id if predicted_outcome == 1 else sinner_id
            loser_id = sinner_id if predicted_outcome == 1 else alcaraz_id
            print(f"\nFinal Prediction: Player {winner_id} is predicted to win against Player {loser_id}.")
    else:
        print("Cannot run example prediction: data, model, or scaler not loaded.")






