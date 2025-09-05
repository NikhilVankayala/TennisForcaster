import os
import sys
import random
from datetime import date, timedelta

# Add the parent directory to the system path to import tennis_predictor
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)

try:
    # Import the prediction function from your tennis_predictor.py
    from tennis_predictor import predict_match_outcome
except ImportError:
    print("Error: Could not import predict_match_outcome from tennis_predictor.py.")
    print("Please ensure tennis_predictor.py is in the same directory or its path is correctly set.")
    sys.exit(1)

# --- Configuration for the Tournament ---
TOURNAMENT_SIZE = 16
TOURNAMENT_NAME = "Mock Wimbledon 2025"
TOURNAMENT_SURFACE = "Grass"  # Important for surface-specific features
TOURNAMENT_BEST_OF = 3  # Best of 3 sets (5 for Grand Slams if your model supports it)
START_DATE = date(2025, 7, 1)

# --- Mock Player Data ---
# These are actual ATP player IDs that should work with your historical data
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
    200055: "Felix Auger-Aliassime",  # Updated ID
}

# Verify we have enough players
if len(player_data) < TOURNAMENT_SIZE:
    print(f"Warning: Only {len(player_data)} players provided for {TOURNAMENT_SIZE} player tournament.")
    print("Adding generic players to fill the tournament...")
    current_max_id = max(player_data.keys()) if player_data else 900000
    for i in range(TOURNAMENT_SIZE - len(player_data)):
        current_max_id += 1
        player_data[current_max_id] = f"Generic Player {current_max_id}"

# Convert to list and randomize draw
players_list = list(player_data.items())
random.shuffle(players_list)

def get_round_name(num_players):
    """Get the proper round name based on number of remaining players."""
    if num_players == 2:
        return "Final"
    elif num_players == 4:
        return "Semi-Finals"
    elif num_players == 8:
        return "Quarter-Finals"
    elif num_players == 16:
        return "Round of 16"
    elif num_players == 32:
        return "Round of 32"
    elif num_players == 64:
        return "Round of 64"
    elif num_players == 128:
        return "First Round"
    else:
        # Calculate round number from the back
        import math
        round_from_final = int(math.log2(num_players))
        return f"Round of {num_players}"

def simulate_round(matches_in_round, current_date, round_name):
    """Simulates matches for a single round and returns the winners."""
    print(f"\n{'='*60}")
    print(f"{round_name} - {current_date.strftime('%B %d, %Y')}")
    print(f"{len(matches_in_round)} matches on {TOURNAMENT_SURFACE} surface")
    print(f"{'='*60}")
    
    winners_of_round = []
    
    for match_num, (p1_id, p1_name, p2_id, p2_name) in enumerate(matches_in_round, 1):
        print(f"\nMatch {match_num}: {p1_name} vs {p2_name}")
        print(f"  Player IDs: {p1_id} vs {p2_id}")
        
        try:
            # Call prediction function with surface and best_of parameters
            prediction_result, win_probability = predict_match_outcome(
                p1_id, p2_id, current_date, 
                surface=TOURNAMENT_SURFACE, 
                best_of=TOURNAMENT_BEST_OF
            )
            
            if prediction_result is None or win_probability is None:
                print(f"  ⚠️  Prediction failed - using random selection")
                winner_id, winner_name = random.choice([(p1_id, p1_name), (p2_id, p2_name)])
                loser_id, loser_name = (p2_id, p2_name) if winner_id == p1_id else (p1_id, p1_name)
            else:
                # prediction_result: 1 if P1 wins, 0 if P2 wins
                # win_probability: probability that P1 wins
                if prediction_result == 1:
                    winner_id, winner_name = p1_id, p1_name
                    loser_id, loser_name = p2_id, p2_name
                else:
                    winner_id, winner_name = p2_id, p2_name
                    loser_id, loser_name = p1_id, p1_name
                
                p1_win_prob = win_probability
                p2_win_prob = 1 - win_probability
                
                print(f"  📊 Win Probabilities:")
                print(f"    {p1_name}: {p1_win_prob:.1%}")
                print(f"    {p2_name}: {p2_win_prob:.1%}")
            
            print(f"  🏆 Winner: {winner_name}")
            winners_of_round.append((winner_id, winner_name))
            
        except Exception as e:
            print(f"  ❌ Error during prediction: {e}")
            print(f"  Using random selection as fallback...")
            winner_id, winner_name = random.choice([(p1_id, p1_name), (p2_id, p2_name)])
            winners_of_round.append((winner_id, winner_name))
            print(f"  🎲 Random Winner: {winner_name}")
    
    return winners_of_round

def print_tournament_bracket(current_players, round_name):
    """Print the current tournament bracket."""
    print(f"\n{round_name} Bracket:")
    print("-" * 40)
    for i in range(0, len(current_players), 2):
        if i + 1 < len(current_players):
            p1_name = current_players[i][1]
            p2_name = current_players[i+1][1]
            print(f"  {p1_name} vs {p2_name}")
    print("-" * 40)

def run_tournament():
    """Runs the full tournament simulation."""
    if not players_list:
        print("❌ No players available for the tournament. Please populate player_data.")
        return
    
    # Ensure we have exactly the right number of players (power of 2)
    if TOURNAMENT_SIZE & (TOURNAMENT_SIZE - 1) != 0:
        print(f"⚠️  Tournament size ({TOURNAMENT_SIZE}) is not a power of 2.")
        print("Tournament brackets work best with 2, 4, 8, 16, 32, 64, 128 players.")
    
    current_players = players_list[:TOURNAMENT_SIZE]
    current_date = START_DATE
    
    print(f"\n🎾 {TOURNAMENT_NAME}")
    print(f"📅 Start Date: {START_DATE.strftime('%B %d, %Y')}")
    print(f"🏟️  Surface: {TOURNAMENT_SURFACE}")
    print(f"👥 Players: {len(current_players)}")
    print(f"🏆 Format: Best of {TOURNAMENT_BEST_OF}")
    
    # Print initial bracket
    print(f"\n🗓️  Tournament Draw:")
    print("=" * 50)
    for i, (player_id, player_name) in enumerate(current_players, 1):
        print(f"{i:2d}. {player_name} (ID: {player_id})")
    print("=" * 50)
    
    round_number = 1
    tournament_results = []
    
    while len(current_players) > 1:
        round_name = get_round_name(len(current_players))
        
        # Print bracket for this round
        print_tournament_bracket(current_players, round_name)
        
        # Create matches for this round
        matches_for_round = []
        for i in range(0, len(current_players), 2):
            if i + 1 < len(current_players):
                player1 = current_players[i]
                player2 = current_players[i + 1]
                matches_for_round.append((player1[0], player1[1], player2[0], player2[1]))
        
        # Handle odd number of players (bye)
        if len(current_players) % 2 == 1:
            bye_player = current_players[-1]
            print(f"\n🎫 {bye_player[1]} receives a bye to the next round")
            winners = simulate_round(matches_for_round, current_date, round_name)
            winners.append(bye_player)
        else:
            winners = simulate_round(matches_for_round, current_date, round_name)
        
        # Store round results
        round_result = {
            'round': round_name,
            'date': current_date,
            'matches': matches_for_round,
            'winners': winners[:]
        }
        tournament_results.append(round_result)
        
        # Advance to next round
        current_players = winners
        current_date += timedelta(days=2)  # 2 days between rounds
        round_number += 1
        
        # Print round summary
        print(f"\n📋 {round_name} Results:")
        for winner_id, winner_name in winners:
            print(f"  ✅ {winner_name}")
    
    # Tournament complete
    if current_players:
        champion_id, champion_name = current_players[0]
        print(f"\n🏆 {TOURNAMENT_NAME} CHAMPION: {champion_name} (ID: {champion_id}) 🏆")
        
        # Print tournament summary
        print(f"\n📊 Tournament Summary:")
        print(f"🗓️  Duration: {START_DATE.strftime('%B %d')} - {(current_date - timedelta(days=2)).strftime('%B %d, %Y')}")
        print(f"🏟️  Surface: {TOURNAMENT_SURFACE}")
        print(f"🎾 Total Rounds: {len(tournament_results)}")
        print(f"👑 Champion: {champion_name}")
        
    else:
        print("\n❌ Tournament ended with no champion (unexpected error).")
    
    return tournament_results

def print_player_path_to_title(champion_name, tournament_results):
    """Print the path the champion took to win the tournament."""
    print(f"\n🛤️  {champion_name}'s Path to Victory:")
    print("-" * 50)
    
    for round_result in tournament_results:
        round_name = round_result['round']
        matches = round_result['matches']
        winners = round_result['winners']
        
        # Find the match involving the champion
        champion_match = None
        for match in matches:
            p1_id, p1_name, p2_id, p2_name = match
            if p1_name == champion_name or p2_name == champion_name:
                opponent = p2_name if p1_name == champion_name else p1_name
                champion_match = f"def. {opponent}"
                break
        
        if champion_match:
            print(f"  {round_name}: {champion_match}")

if __name__ == "__main__":
    print("🎾 Tennis Tournament Simulator")
    print("=" * 40)
    
    try:
        results = run_tournament()
        
        if results and len(results) > 0:
            # Get champion from final results
            final_round = results[-1]
            if final_round['winners']:
                champion_name = final_round['winners'][0][1]
                print_player_path_to_title(champion_name, results)
        
        print(f"\n✅ Tournament simulation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Tournament simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Tournament simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
