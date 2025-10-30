import os
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from supabase import create_client, Client
import json
import math

# --- SERVER CONFIGURATION ---
SCORE_HISTORY_WINDOW = 10
# NOTE: Ensure these environment variables are set correctly on your hosting platform (e.g., Render)
# The server will look for these environment variables (SUPABASE_URL, SUPABASE_KEY).
SUPABASE_URL = os.environ.get("SUPABASE_URL", "DEFAULT_SUPABASE_URL") 
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "DEFAULT_SUPABASE_KEY")
SUPABASE_TABLE = "scores_history"
DIFFICULTY_LEVELS = 3
DEFAULT_DIFFICULTY_INDEX = 1

# Initialize Supabase Client
try:
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "DEFAULT_SUPABASE_URL":
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized.")
    else:
        print("SUPABASE_URL or SUPABASE_KEY not set. Supabase integration disabled.")
        supabase = None
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    supabase = None

# --- IMPORTANT RLS/PERMISSIONS NOTE ---
# The error "violates row-level security policy" means the Supabase database
# is rejecting the insertion, NOT that the Python code is broken.
# TO FIX THIS: You must go to your Supabase project and adjust the RLS policy
# for the 'scores_history' table to allow the key used by this server (e.g., the Anon key)
# to perform INSERT operations.
# ---------------------------------------

# --- RL CONFIGURATION (Q-Learning) ---
# Environment parameters (Tuning these values can drastically change the AI's learning behavior)
RL_Y_BUCKETS = 10
RL_VEL_BUCKETS = 8
RL_DIST_BUCKETS = 8
RL_GAP_BUCKETS = 5
ACTIONS = 2
SCREEN_HEIGHT = 64
SCREEN_WIDTH = 128
STATES = RL_Y_BUCKETS * RL_VEL_BUCKETS * RL_DIST_BUCKETS * RL_GAP_BUCKETS

# RL Hyperparameters
ALPHA = 0.25  # Learning rate
GAMMA = 0.90  # Discount factor
EPSILON = 0.30 # Exploration rate (Epsilon-greedy)

# Initialize Q-Table (In-Memory)
Q = np.zeros((STATES, ACTIONS), dtype=np.float32)
try:
    # Attempt to load saved Q-table for persistence
    Q = np.load("qtable.npy")
    print("Q-table loaded successfully from qtable.npy.")
except FileNotFoundError:
    print("qtable.npy not found. Initializing Q-table to zeros.")

# --- DYNAMIC DIFFICULTY/ANTAGONIST CONFIGURATION ---

# Base Difficulty Parameters (Matches the HARD setting on the Arduino - Index 2)
BASE_PIPE_SPEED = 2.9
BASE_GAP_HEIGHT = 20.0 
BASE_GRAVITY = 0.50

# Difficulty Limits (Clamping values for the antagonist parameters)
MAX_PIPE_SPEED = 4.5
MIN_PIPE_SPEED = 2.0 # Minimum speed allowed
MIN_GAP_HEIGHT = 15.0  # Smallest gap (hardest)
MAX_GAP_HEIGHT = 30.0  # Largest gap (easiest)
MAX_GRAVITY = 0.65
MIN_GRAVITY = 0.40 # Minimum gravity allowed

# Adjustment Steps (Tuned for gradual change over a few successful/failed attempts)
# Positive reward (doing well, cleared 3 pipes) -> Increase difficulty
POS_REWARD_STEP_SPEED = 0.05
POS_REWARD_STEP_GAP = -1.0     # Decrease gap (Harder)
POS_REWARD_STEP_GRAVITY = 0.02 # Increase gravity (Harder)

# Negative reward (crashed) -> Decrease difficulty
NEG_REWARD_STEP_SPEED = -0.10  # Larger decrease
NEG_REWARD_STEP_GAP = 2.0      # Increase gap (Easier)
NEG_REWARD_STEP_GRAVITY = -0.04

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
    reward: float 
    done: bool
    player_name: str

class AdaptiveScore(BaseModel):
    player: str = Field(alias="player_name")
    score: int

class AntagonistResponse(BaseModel):
    action: int
    pipeSpeed: float
    gapHeight: float 
    gravity: float

# --- RL HELPER FUNCTIONS (Discretization) ---
def get_state_index(birdY: float, birdVel: float, pipeXlocal: float, gapYpos: float) -> int:
    """Discretizes the 4 continuous state variables into a single index."""
    if any(math.isnan(x) for x in [birdY, birdVel, pipeXlocal, gapYpos]):
        return -1 

    # 1. Bird Y Position
    yBucket = int((birdY * RL_Y_BUCKETS) / (SCREEN_HEIGHT + 1))
    yBucket = max(0, min(RL_Y_BUCKETS - 1, yBucket))

    # 2. Bird Velocity
    vcl = min(max(birdVel, -8.0), 8.0)  # Clamp to a reasonable range
    velBucket = int(((vcl + 8.0) / 16.0) * RL_VEL_BUCKETS) 
    velBucket = max(0, min(RL_VEL_BUCKETS - 1, velBucket))

    # 3. Pipe Distance
    dist = max(0, min(pipeXlocal, SCREEN_WIDTH))
    distBucket = int((dist * RL_DIST_BUCKETS) / (SCREEN_WIDTH + 1))
    distBucket = max(0, min(RL_DIST_BUCKETS - 1, distBucket))
    
    # 4. Gap Position
    clamped_gapYpos = max(0, min(gapYpos, SCREEN_HEIGHT))
    gapBucket = int((clamped_gapYpos * RL_GAP_BUCKETS) / (SCREEN_HEIGHT + 1))
    gapBucket = max(0, min(RL_GAP_BUCKETS - 1, gapBucket))

    # Combine buckets into a single index
    idx = yBucket
    idx = idx * RL_VEL_BUCKETS + velBucket
    idx = idx * RL_DIST_BUCKETS + distBucket
    idx = idx * RL_GAP_BUCKETS + gapBucket
    
    idx = max(0, min(STATES - 1, idx))
    
    return idx

# --- NEW ANTAGONIST LOGIC ---
def adjust_antagonist_params(player_name: str, reward: float, is_done: bool) -> Dict[str, Any]:
    """
    Adjusts game parameters based on accumulated reward.
    Returns the *new target* parameters for the client.
    """
    if player_name not in antagonist_states:
        antagonist_states[player_name] = get_initial_antagonist_params()

    current_params = antagonist_states[player_name]
    
    # If the game just ended, reset to base difficulty for the next round
    if is_done:
        # Player crashed (reward is -1.0), reset difficulty to base (HARD)
        antagonist_states[player_name] = get_initial_antagonist_params()
        return antagonist_states[player_name]

    adj_speed = 0.0
    adj_gap = 0.0
    adj_gravity = 0.0
    
    # Successful step (reward > 0.0) -> Increase Difficulty
    if reward > 0.0:
        adj_speed = POS_REWARD_STEP_SPEED
        adj_gap = POS_REWARD_STEP_GAP
        adj_gravity = POS_REWARD_STEP_GRAVITY
        
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

# --- RL ENDPOINT (COMPETITOR MODE) ---
@app.post("/api/rl-step", response_model=AntagonistResponse)
async def rl_step(experience: RlExperience):
    """
    Receives experience packet, updates Q-table, adjusts antagonist params,
    and returns next action and new difficulty parameters.
    """
    global Q
    s = get_state_index(*experience.state)
    a = experience.action
    r = experience.reward
    done = experience.done
    player_name = experience.player_name

    # Handle invalid state index
    if s < 0 or s >= STATES or a < 0 or a >= ACTIONS:
        print(f"Warning: Invalid state or action index received. s={s}, a={a}. Using default/current params.")
        current_params = antagonist_states.get(player_name, get_initial_antagonist_params())
        return AntagonistResponse(
            action=np.random.randint(ACTIONS), 
            **current_params
        )

    # --- Q-Learning Update ---
    if done:
        target = r
        # When done, next action doesn't matter, but we pick one anyway.
        next_action_index = 1 # Default to no-op for simplicity
    else:
        # Standard Q-learning equation uses the MAX Q-value of the current state 's'
        # to calculate the target (since we don't know the next state 's' prime)
        max_q_s = np.max(Q[s]) 
        target = r + GAMMA * max_q_s
        
        # Epsilon-Greedy Policy for next action selection based on state 's'
        if np.random.rand() < EPSILON:
            next_action_index = np.random.randint(ACTIONS) # Explore
        else:
            next_action_index = np.argmax(Q[s]) # Exploit

    # Update Q-table value for the state-action pair (s, a)
    Q[s, a] = Q[s, a] + ALPHA * (target - Q[s, a])

    # --- Adjust Antagonist/Difficulty Parameters ---
    new_antagonist_params = adjust_antagonist_params(player_name, r, done)

    # --- Return Response ---
    return AntagonistResponse(
        action=int(next_action_index),
        **new_antagonist_params
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
    
    new_difficulty = determine_adaptive_difficulty(data.score)

    # Log to Supabase if available
    global supabase
    if supabase:
        try:
            # This logic is compatible with the scores_history.sql schema in the Canvas
            supabase.table(SUPABASE_TABLE).insert({
                "player_name": data.player, 
                "score": data.score,
                "flaps": 0,
                "difficulty_index": new_difficulty
            }).execute()
        except Exception as e:
            print(f"Supabase Error during adaptive log: {e}")

    # Return the difficulty index for the client to use
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
        "current_antagonist_states": antagonist_states 
    }

@app.get("/api/save_q")
async def save_q_table():
    """Manual endpoint to save the Q-table."""
    try:
        np.save("qtable.npy", Q)
        return {"message": "Q-table saved successfully to qtable.npy"}
    except Exception as e:
        return {"error": f"Failed to save Q-table: {e}"}
