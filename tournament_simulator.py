import os
import sys
import random
from datetime import date, timedelta

# Add the parent directory to the system path to import tennis_predictor
# This assumes tournament_simulator.py is at the same level as tennis_predictor.py
# If tennis_predictor.py is in a 'scripts' folder, adjust this path accordingly.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir)) # Assuming tennis_predictor.py is in project root
sys.path.insert(0, project_root)

try:
    # Import the prediction function from your tennis_predictor.py
    from tennis_predictor import predict_match_outcome
except ImportError:
    print("Error: Could not import predict_match_outcome from tennis_predictor.py.")
    print("Please ensure tennis_predictor.py is in the same directory or its path is correctly set.")
    sys.exit(1)

# --- Configuration for the Tournament ---
TOURNAMENT_SIZE = 16 # Changed from 128 to 16
TOURNAMENT_NAME = "Mock Wimbledon 2025"
TOURNAMENT_SURFACE = "Grass" # Important for surface-specific features
START_DATE = date(2025, 7, 1) # Start date for the tournament simulation

# --- Mock Player Data (REPLACE WITH YOUR ACTUAL PLAYER IDs AND NAMES) ---
# IMPORTANT: These are placeholder IDs. For accurate results, replace these
# with actual player_id values found in your player_df_historical.joblib.
# You can get a list of player IDs from your raw ATP data files.
# Example: Novak Djokovic (104925), Carlos Alcaraz (200780), Jannik Sinner (200000 or 105357)
# You need 16 unique player IDs for a 16-man tournament.
player_data = {
    104925: "Novak Djokovic",
    207989: "Carlos Alcaraz",
    206173: "Jannik Sinner",
    106421: "Daniil Medvedev",
    100644: "Alexander Zverev",
    126094: "Andrey Rublev",
    134770: "Casper Ruud",
    128034: "Hubert Hurkacz",
    105777: "Grigor Dimitrov",
    126203: "Taylor Fritz",
    200282: "Alex de Minaur",
    126205: "Tommy Paul",
    210097: "Ben Shelton",
    126207: "Frances Tiafoe",
    208029: "Holger Rune",
    200000: "Felix Auger-Aliassime",
    # If TOURNAMENT_SIZE is 16, this list is now sufficient.
    # If you reduce TOURNAMENT_SIZE further, you can remove players from this list.
}

# Auto-fill remaining players if less than TOURNAMENT_SIZE are provided
if len(player_data) < TOURNAMENT_SIZE:
    print(f"Warning: Only {len(player_data)} players provided. Auto-filling with generic players.")
    current_max_id = max(player_data.keys()) if player_data else 900000 # Start IDs higher for generics
    for i in range(TOURNAMENT_SIZE - len(player_data)):
        current_max_id += 1
        player_data[current_max_id] = f"Generic Player {current_max_id}"

# Convert to a list of (ID, Name) tuples for shuffling
players_list = list(player_data.items())
random.shuffle(players_list) # Randomize seeding for the draw

# --- Tournament Simulation Logic ---

def simulate_round(matches_in_round, current_date, round_name):
    """Simulates matches for a single round and returns the winners."""
    print(f"\n--- {round_name} ({len(matches_in_round)} Matches) - Date: {current_date.strftime('%Y-%m-%d')} ---")
    winners_of_round = []
    for i, (p1_id, p1_name, p2_id, p2_name) in enumerate(matches_in_round):
        print(f"\nMatch {i+1}: {p1_name} (ID: {p1_id}) vs {p2_name} (ID: {p2_id})")
        # Call your predict_match_outcome function
        # Note: tourney_date is passed as a datetime.date object, which predict_match_outcome converts to Timestamp
        prediction_result, probabilities = predict_match_outcome(p1_id, p2_id, current_date)

        if prediction_result is None:
            print(f"  Prediction failed for {p1_name} vs {p2_name}. Skipping match.")
            # Simple fallback: randomly pick a winner if prediction fails
            winner_id, winner_name = random.choice([(p1_id, p1_name), (p2_id, p2_name)])
            print(f"  (Fallback: Randomly selected {winner_name} as winner)")
        else:
            if prediction_result == 1:
                winner_id, winner_name = p1_id, p1_name
            else:
                winner_id, winner_name = p2_id, p2_name
            print(f"  Predicted Winner: {winner_name} (P1 Win Prob: {probabilities[1]:.2f}, P2 Win Prob: {probabilities[0]:.2f})")

        winners_of_round.append((winner_id, winner_name))
    return winners_of_round

def run_tournament():
    """Runs the full tournament simulation."""
    if not players_list:
        print("No players available for the tournament. Please populate player_data.")
        return

    current_players = players_list[:] # Start with all players
    current_date = START_DATE
    round_num = 1

    print(f"\n--- Starting {TOURNAMENT_NAME} ({TOURNAMENT_SIZE} Players) ---")

    while len(current_players) > 1:
        round_name = ""
        # Adjusted round names for a smaller tournament if needed, but general logic holds
        if len(current_players) == 16: round_name = "Round of 16"
        elif len(current_players) == 8: round_name = "Quarter-Finals"
        elif len(current_players) == 4: round_name = "Semi-Finals"
        elif len(current_players) == 2: round_name = "Final"
        else: round_name = f"Round {round_num}" # Fallback for other sizes

        # Pair players for the current round
        matches_for_round = []
        for i in range(0, len(current_players), 2):
            player1 = current_players[i]
            player2 = current_players[i+1]
            matches_for_round.append((player1[0], player1[1], player2[0], player2[1])) # (id1, name1, id2, name2)

        # Simulate the round
        winners = simulate_round(matches_for_round, current_date, round_name)
        current_players = winners
        current_date += timedelta(days=1) # Advance date for next round
        round_num += 1

    if current_players:
        champion_id, champion_name = current_players[0]
        print(f"\n--- {TOURNAMENT_NAME} Champion: {champion_name} (ID: {champion_id}) ---")
    else:
        print("\nTournament ended with no champion (unexpected error).")

if __name__ == "__main__":
    run_tournament()
