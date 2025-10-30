import os
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from supabase import create_client, Client
import json
import math # Used for isnan checks later if needed

# --- SERVER CONFIGURATION ---
SCORE_HISTORY_WINDOW = 10
SUPABASE_URL = os.environ.get("SUPABASE_URL", "DEFAULT_SUPABASE_URL") # Provide a default or handle missing env var
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "DEFAULT_SUPABASE_KEY") # Provide a default or handle missing env var
SUPABASE_TABLE = "scores_history"
DIFFICULTY_LEVELS = 3
DEFAULT_DIFFICULTY_INDEX = 1

# Initialize Supabase Client
try:
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "DEFAULT_SUPABASE_URL":
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized.")
    else:
        print("SupABASE_URL or SUPABASE_KEY not set. Supabase integration disabled.")
        supabase = None
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    supabase = None

# --- RL CONFIGURATION (Q-Learning) ---
# Environment parameters
RL_Y_BUCKETS = 10
RL_VEL_BUCKETS = 8
RL_DIST_BUCKETS = 8
RL_GAP_BUCKETS = 5
ACTIONS = 2
SCREEN_HEIGHT = 64
SCREEN_WIDTH = 128
STATES = RL_Y_BUCKETS * RL_VEL_BUCKETS * RL_DIST_BUCKETS * RL_GAP_BUCKETS

# RL Hyperparameters
ALPHA = 0.25  
GAMMA = 0.90  
EPSILON = 0.30  

# Initialize Q-Table (In-Memory)
Q = np.zeros((STATES, ACTIONS), dtype=np.float32)
try:
    Q = np.load("qtable.npy")
    print("Q-table loaded successfully from qtable.npy.")
except FileNotFoundError:
    print("qtable.npy not found. Initializing Q-table to zeros.")

# --- DYNAMIC DIFFICULTY/ANTAGONIST CONFIGURATION ---

# Base Difficulty Parameters (Matches Arduino's HARD setting - Index 2)
BASE_PIPE_SPEED = 2.9
BASE_GAP_HEIGHT = 20.0 # Use float for calculations
BASE_GRAVITY = 0.50

# Difficulty Limits
MAX_PIPE_SPEED = 4.5
MIN_PIPE_SPEED = 2.0 # Minimum speed allowed
MIN_GAP_HEIGHT = 15.0 
MAX_GAP_HEIGHT = 30.0 # Largest gap (easiest)
MAX_GRAVITY = 0.65
MIN_GRAVITY = 0.40 # Minimum gravity allowed

# Adjustment Steps (How much to change per RL update based on reward)
# Positive reward (doing well) -> Increase difficulty
POS_REWARD_STEP_SPEED = 0.03
POS_REWARD_STEP_GAP = -0.5  # Decrease gap
POS_REWARD_STEP_GRAVITY = 0.01

# Negative reward (doing poorly) -> Decrease difficulty
NEG_REWARD_STEP_SPEED = -0.06 # Larger decrease
NEG_REWARD_STEP_GAP = 1.0   # Increase gap
NEG_REWARD_STEP_GRAVITY = -0.02

# Player Antagonist State Storage
antagonist_states: Dict[str, Dict[str, Any]] = {}

def get_initial_antagonist_params() -> Dict[str, Any]:
    """Returns the base starting parameters for a player."""
    return {
        "pipeSpeed": BASE_PIPE_SPEED,
        "gapHeight": BASE_GAP_HEIGHT,
        "gravity": BASE_GRAVITY
    }

# --- FASTAPI SETUP ---
app = FastAPI(title="ReflexIQ RL/Antagonist Backend")

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---

class RlExperience(BaseModel):
    state: List[float]
    action: int
    reward: float # Accumulated reward since last send
    done: bool
    player_name: str

class AdaptiveScore(BaseModel):
    player: str = Field(alias="player_name")
    score: int

# **UPDATED** Response Model for Competitor Mode
class AntagonistResponse(BaseModel):
    action: int
    pipeSpeed: float
    gapHeight: float # Send as float
    gravity: float

# --- RL HELPER FUNCTIONS (Discretization - Unchanged) ---
def get_state_index(birdY: float, birdVel: float, pipeXlocal: float, gapYpos: float) -> int:
    # Ensure inputs are valid numbers
    if any(math.isnan(x) for x in [birdY, birdVel, pipeXlocal, gapYpos]):
        return -1 # Indicate invalid state

    yBucket = int((birdY * RL_Y_BUCKETS) / (SCREEN_HEIGHT + 1))
    yBucket = max(0, min(RL_Y_BUCKETS - 1, yBucket))

    # Clamp velocity more realistically if needed
    vcl = min(max(birdVel, -8.0), 8.0) 
    # Adjusted velocity bucket calculation for potentially wider range
    velBucket = int(((vcl + 8.0) / 16.0) * RL_VEL_BUCKETS) 
    velBucket = max(0, min(RL_VEL_BUCKETS - 1, velBucket))

    # Clamp distance
    dist = max(0, min(pipeXlocal, SCREEN_WIDTH))
    distBucket = int((dist * RL_DIST_BUCKETS) / (SCREEN_WIDTH + 1))
    distBucket = max(0, min(RL_DIST_BUCKETS - 1, distBucket))
    
    # Clamp gap position
    clamped_gapYpos = max(0, min(gapYpos, SCREEN_HEIGHT))
    gapBucket = int((clamped_gapYpos * RL_GAP_BUCKETS) / (SCREEN_HEIGHT + 1))
    gapBucket = max(0, min(RL_GAP_BUCKETS - 1, gapBucket))

    # Calculate index safely
    idx = yBucket
    idx = idx * RL_VEL_BUCKETS + velBucket
    idx = idx * RL_DIST_BUCKETS + distBucket
    idx = idx * RL_GAP_BUCKETS + gapBucket
    
    # Final check for index bounds
    idx = max(0, min(STATES - 1, idx))
    
    return idx

# --- ANTAGONIST LOGIC (Unchanged) ---
def adjust_antagonist_params(player_name: str, reward: float, is_done: bool) -> Dict[str, Any]:
    """
    Adjusts game parameters based on accumulated reward.
    Returns the *new target* parameters for the client.
    """
    if player_name not in antagonist_states:
        antagonist_states[player_name] = get_initial_antagonist_params()

    # Get current parameters for this player
    current_params = antagonist_states[player_name]
    
    # If the game just ended, reset to base difficulty for the next round
    if is_done:
        antagonist_states[player_name] = get_initial_antagonist_params()
        return antagonist_states[player_name] # Return base params immediately

    adj_speed = 0.0
    adj_gap = 0.0
    adj_gravity = 0.0
    
    # Thresholds for reward interpretation (tune these)
    GOOD_PERFORMANCE_THRESHOLD = 0.5 # e.g., survived a while or cleared a pipe recently
    POOR_PERFORMANCE_THRESHOLD = -0.5 # e.g., crashed

    # Positive Reward -> Increase Difficulty
    if reward > GOOD_PERFORMANCE_THRESHOLD:
        adj_speed = POS_REWARD_STEP_SPEED
        adj_gap = POS_REWARD_STEP_GAP
        adj_gravity = POS_REWARD_STEP_GRAVITY
        
    # Negative Reward -> Decrease Difficulty
    elif reward < POOR_PERFORMANCE_THRESHOLD: 
        adj_speed = NEG_REWARD_STEP_SPEED
        adj_gap = NEG_REWARD_STEP_GAP
        adj_gravity = NEG_REWARD_STEP_GRAVITY

    # Apply adjustments and clamp within defined limits
    new_speed = max(MIN_PIPE_SPEED, min(MAX_PIPE_SPEED, current_params['pipeSpeed'] + adj_speed))
    new_gap = max(MIN_GAP_HEIGHT, min(MAX_GAP_HEIGHT, current_params['gapHeight'] + adj_gap))
    new_gravity = max(MIN_GRAVITY, min(MAX_GRAVITY, current_params['gravity'] + adj_gravity))

    # Update the player's state
    current_params['pipeSpeed'] = new_speed
    current_params['gapHeight'] = new_gap
    current_params['gravity'] = new_gravity

    antagonist_states[player_name] = current_params
    return current_params

# --- RL ENDPOINT (COMPETITOR MODE - Unchanged) ---
@app.post("/api/rl-step", response_model=AntagonistResponse)
async def rl_step(experience: RlExperience):
    """
    Receives experience, updates Q-table, adjusts antagonist params, 
    returns next action and new params.
    """
    global Q, antagonist_states

    # 1. Get current state index (s)
    s = get_state_index(*experience.state)
    
    a = experience.action
    r = experience.reward # Accumulated reward
    done = experience.done
    player_name = experience.player_name

    # Handle invalid state index gracefully
    if s < 0 or s >= STATES or a < 0 or a >= ACTIONS:
        print(f"Warning: Invalid state or action index received. s={s}, a={a}. State: {experience.state}")
        # Return current params and a random action if state is invalid
        current_params = antagonist_states.get(player_name, get_initial_antagonist_params())
        return AntagonistResponse(
            action=np.random.randint(ACTIONS), 
            pipeSpeed=current_params['pipeSpeed'],
            gapHeight=current_params['gapHeight'],
            gravity=current_params['gravity']
        )

    # --- Q-Learning Update ---
    if done:
        target = r
        next_action_index = np.random.randint(ACTIONS) # Action for next state (which won't happen)
    else:
        max_q_s = np.max(Q[s]) # Use max Q of current state 's'
        target = r + GAMMA * max_q_s
        
        # Epsilon-Greedy Policy for next action selection based on state 's'
        if np.random.rand() < EPSILON:
            next_action_index = np.random.randint(ACTIONS) # Explore
        else:
            next_action_index = np.argmax(Q[s]) # Exploit

    # Update Q-table value for the state-action pair (s, a)
    Q[s, a] = Q[s, a] + ALPHA * (target - Q[s, a])

    # --- Adjust Antagonist/Difficulty Parameters ---
    # This uses the reward from the *just completed* step to decide the *next* difficulty
    new_antagonist_params = adjust_antagonist_params(player_name, r, done)

    # --- Return Response ---
    return AntagonistResponse(
        action=int(next_action_index),
        pipeSpeed=new_antagonist_params['pipeSpeed'],
        gapHeight=new_antagonist_params['gapHeight'], # Send float
        gravity=new_antagonist_params['gravity']
    )


# --- ADAPTIVE ENDPOINT (FIXED) ---

def determine_adaptive_difficulty(score: int) -> int:
    """
    Determines the difficulty index (0, 1, or 2) based on the player's current score.
    This provides immediate feedback to the player in Adaptive mode.
    """
    if score < 10:
        return 0  # EASY (Difficulty 0: Easiest gravity, largest gap)
    elif score >= 30:
        return 2  # HARD (Difficulty 2: Hardest gravity, smallest gap)
    else:
        return 1  # MEDIUM (Difficulty 1: Default balance)

@app.post("/adaptive_logic")
async def adaptive_logic(data: AdaptiveScore):
    """Logs score history and returns an adaptive difficulty index (0, 1, or 2)."""
    
    # 1. Determine new difficulty based on the score received
    new_difficulty = determine_adaptive_difficulty(data.score)

    # 2. Log to Supabase if available
    global supabase
    if supabase:
        try:
            # This logic is compatible with the scores_history.sql schema (which includes 'difficulty_index')
            supabase.table(SUPABASE_TABLE).insert({
                "player_name": data.player, 
                "score": data.score,
                "flaps": 0, # Placeholder, as discussed
                "difficulty_index": new_difficulty # Log the calculated difficulty
            }).execute()
        except Exception as e:
            print(f"Supabase Error during adaptive log: {e}")

    # 3. Return the difficulty index for the client to use
    return {"player": data.player, "difficulty": new_difficulty}


# --- UTILITY ENDPOINTS (Unchanged) ---
@app.get("/api/info")
async def get_info():
    """Returns RL parameters and Q-table summary."""
    return {
        "status": "RL Server Active",
        "state_space_size": STATES,
        "action_space_size": ACTIONS,
        "q_table_sum": float(np.sum(Q)),
        "q_table_max": float(np.max(Q)),
        "epsilon": EPSILON, "gamma": GAMMA, "alpha": ALPHA,
        "current_antagonist_states": antagonist_states # Show current dynamic difficulties
    }

@app.get("/api/save_q")
async def save_q_table():
    """Manual endpoint to save the Q-table."""
    try:
        np.save("qtable.npy", Q)
        return {"message": "Q-table saved successfully to qtable.npy"}
    except Exception as e:
        return {"error": f"Failed to save Q-table: {e}"}

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    import uvicorn
    print("Starting FastAPI server locally on http://0.0.0.0:8000")
    # Use reload=True for development, consider removing for production
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

