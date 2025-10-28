import os
import json
import numpy as np
from flask import Flask, request, jsonify
from supabase import create_client, Client 

# --- SERVER CONFIGURATION ---
SCORE_HISTORY_WINDOW = 10
DIFFICULTY_LEVELS = 3
DEFAULT_DIFFICULTY_INDEX = 1

# Environment variables are critical for security and deployment (e.g., on Render)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_KEY")
SUPABASE_TABLE = "scores_history" 

# Initialize Supabase Client
try:
    # IMPORTANT: Ensure SUPABASE_URL and SUPABASE_KEY are set
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase client. Check environment variables: {e}")
    supabase = None 

# --- ADAPTIVE LOGIC FUNCTIONS (Mirrors C++ Complexity) ---

def compute_trend(arr):
    """Compute linear trend (slope) for a score list for performance analysis."""
    n = len(arr)
    if n <= 1: return 0.0
    x = np.arange(n)
    y = np.array(arr)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denom = (n * sum_x2 - sum_x * sum_x)
    
    if abs(denom) < 1e-6: return 0.0
        
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return float(slope)

def compute_std_dev(arr):
    """Compute standard deviation of a list for consistency analysis."""
    if not arr: return 0.0
    return float(np.std(arr))

def compute_adaptive_difficulty(score_hist, flap_hist):
    """
    Computes the suggested adaptive difficulty index (0-2) using the complex 
    formula involving trend, consistency, average score, and normalized flap rate.
    """
    
    # 1. Calculate Score Metrics
    avg = np.mean(score_hist) if score_hist else 0
    stddev = compute_std_dev(score_hist)
    trend = compute_trend(score_hist) 

    # 2. Calculate Flap Metrics
    avg_flaps = np.mean(flap_hist) if flap_hist else 0

    # Flap factor: rewards high activity relative to score, weighted by 0.5.
    # This detects high engagement/effort, suggesting ability to handle higher difficulty.
    flap_factor = (avg_flaps / (avg + 1.0)) * 0.5
    
    # 3. Improvement Factor (Matching C++ Formula)
    improvement_factor = trend * 2.0 - (stddev / 6.0) + (avg / 30.0) + flap_factor

    # 4. Inertia and Clamping Logic (Matching C++ Thresholds)
    candidate = DEFAULT_DIFFICULTY_INDEX
    
    if improvement_factor > 1.4:
        candidate = 2
    elif improvement_factor < -1.0:
        candidate = 0
    elif improvement_factor > 0.8:
        candidate = 2 
    elif improvement_factor < -0.6:
        candidate = 0
    
    # The server calculates a strong suggestion, which the ESP32 then dampens with its local inertia.
    return max(0, min(DIFFICULTY_LEVELS - 1, candidate))

# --- FLASK APPLICATION SETUP ---

app = Flask(__name__)

@app.route('/adaptive_logic', methods=['POST'])
def adaptive_logic():
    global supabase
    if not supabase:
        print("Error: Supabase client not initialized.")
        return jsonify({"error": "Database service unavailable"}), 503

    try:
        data = request.get_json(force=True)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400

    player_name = data.get('player')
    score = data.get('score')
    # Assumes ESP32 sends "flaps" in the POST request body
    flaps = data.get('flaps', 0) 

    if player_name is None or score is None:
        return jsonify({"error": "Missing 'player' or 'score' in request body"}), 400
        
    try:
        score = int(score)
        flaps = int(flaps)
    except ValueError:
        return jsonify({"error": "Score/Flaps must be integers"}), 400
        
    try:
        # 1. LOG NEW SCORE (INSERT)
        supabase.table(SUPABASE_TABLE).insert({
            "player_name": player_name, 
            "score": score,
            "flaps": flaps
        }).execute()

        # 2. FETCH HISTORY (SELECT)
        response = supabase.table(SUPABASE_TABLE).select("score, flaps").eq(
            "player_name", player_name
        ).order(
            "created_at", desc=True
        ).limit(
            SCORE_HISTORY_WINDOW
        ).execute()
        
        history_data = response.data
        if not history_data:
            return jsonify({"player": player_name, "difficulty": DEFAULT_DIFFICULTY_INDEX}), 200

        # Extract and reverse scores/flaps for trend calculation (oldest first)
        score_history = [item['score'] for item in history_data][::-1] 
        flap_history = [item['flaps'] for item in history_data][::-1]

        # 3. COMPUTE DIFFICULTY
        new_difficulty_index = compute_adaptive_difficulty(score_history, flap_history)
        
        return jsonify({"player": player_name, "difficulty": new_difficulty_index}), 200

    except Exception as e:
        print(f"Supabase Error: {e}")
        return jsonify({"error": "Server database error"}), 500

@app.route('/scores', methods=['GET'])
def get_scores():
    """Simple GET endpoint to view the latest scores."""
    global supabase
    if not supabase:
        return jsonify({"error": "Database service unavailable"}), 503
    try:
        response = supabase.table(SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(50).execute()
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": "Server database error"}), 500

# --- SERVER STARTUP ---

if __name__ == '__main__':
    # Running locally uses Flask's built-in server. 
    # Render deployment uses Gunicorn (via the 'gunicorn server:app' command).
    print("Starting Flask server locally on http://0.0.0.0:5000")
    print("Database connection URL:", SUPABASE_URL)
    app.run(host='0.0.0.0', port=5000, debug=True)
