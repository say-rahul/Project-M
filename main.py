import os
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from supabase import create_client, Client
import json

# --- SERVER CONFIGURATION ---
SCORE_HISTORY_WINDOW = 10
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_KEY")
SUPABASE_TABLE = "scores_history" 
DIFFICULTY_LEVELS = 3
DEFAULT_DIFFICULTY_INDEX = 1

# Initialize Supabase Client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    supabase = None 

# --- RL CONFIGURATION (Q-Learning) ---
# Environment parameters (3200 states)
RL_Y_BUCKETS = 10
RL_VEL_BUCKETS = 8
RL_DIST_BUCKETS = 8
RL_GAP_BUCKETS = 5
ACTIONS = 2
SCREEN_HEIGHT = 64
SCREEN_WIDTH = 128
STATES = RL_Y_BUCKETS * RL_VEL_BUCKETS * RL_DIST_BUCKETS * RL_GAP_BUCKETS

# RL Hyperparameters (ADJUSTED FOR FASTER LEARNING/CRASH AVOIDANCE)
ALPHA = 0.25  # Increased Learning Rate (was 0.12)
GAMMA = 0.90  # Reduced Discount Factor (was 0.95)
EPSILON = 0.30 # Increased Exploration Rate (was 0.15)

# Initialize Q-Table (In-Memory)
Q = np.zeros((STATES, ACTIONS), dtype=np.float32)
try:
    Q = np.load("qtable.npy")
    print("Q-table loaded successfully from qtable.npy.")
except FileNotFoundError:
    print("qtable.npy not found. Initializing Q-table to zeros.")

# --- FASTAPI SETUP ---
app = FastAPI(title="ReflexIQ RL/Adaptive Backend")

# Configure CORS (CRITICAL for ESP32/Cross-Origin communication)
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

class ActionResponse(BaseModel):
    action: int

# --- RL HELPER FUNCTIONS (Discretization) ---

def get_state_index(birdY: float, birdVel: float, pipeXlocal: float, gapYpos: float) -> int:
    yBucket = int((birdY * RL_Y_BUCKETS) / (SCREEN_HEIGHT + 1))
    yBucket = max(0, min(RL_Y_BUCKETS - 1, yBucket))

    vcl = min(max(birdVel, -6.0), 6.0)
    velBucket = int(((vcl + 6.0) / 12.0) * RL_VEL_BUCKETS)
    velBucket = max(0, min(RL_VEL_BUCKETS - 1, velBucket))

    dist = max(0, min(pipeXlocal, SCREEN_WIDTH))
    distBucket = int((dist * RL_DIST_BUCKETS) / (SCREEN_WIDTH + 1))
    distBucket = max(0, min(RL_DIST_BUCKETS - 1, distBucket))

    gapBucket = int((gapYpos * RL_GAP_BUCKETS) / (SCREEN_HEIGHT + 1))
    gapBucket = max(0, min(RL_GAP_BUCKETS - 1, gapBucket))

    idx = yBucket
    idx = idx * RL_VEL_BUCKETS + velBucket
    idx = idx * RL_DIST_BUCKETS + distBucket
    idx = idx * RL_GAP_BUCKETS + gapBucket
    
    return idx

# --- RL ENDPOINT (COMPETITOR MODE) ---

@app.post("/api/rl-step", response_model=ActionResponse)
async def rl_step(experience: RlExperience):
    """
    Receives an experience tuple, updates the Q-table, and returns the next action.
    """
    global Q 
    
    s = get_state_index(*experience.state)
    s_prime = s 
    
    a = experience.action
    r = experience.reward
    done = experience.done
    
    if s >= STATES or a >= ACTIONS:
        raise HTTPException(status_code=400, detail="Invalid state or action index.")

    # Q-Learning Update
    if done:
        target = r
    else:
        max_q_s_prime = np.max(Q[s_prime])
        target = r + GAMMA * max_q_s_prime

    Q[s, a] = Q[s, a] + ALPHA * (target - Q[s, a])

    # Epsilon-Greedy Policy for Next Action
    if np.random.rand() < EPSILON and not done:
        next_action = np.random.randint(ACTIONS) # Explore
    else:
        next_action = np.argmax(Q[s_prime]) # Exploit
    
    return ActionResponse(action=int(next_action))


# --- ADAPTIVE ENDPOINT (ADAPTIVE MODE LOGGING) ---

@app.post("/adaptive_logic")
async def adaptive_logic(data: AdaptiveScore):
    """
    Logs score history from the ESP32's local adaptive mode to Supabase.
    """
    global supabase
    if not supabase:
        return {"player": data.player, "difficulty": DEFAULT_DIFFICULTY_INDEX}

    try:
        supabase.table(SUPABASE_TABLE).insert({
            "player_name": data.player, 
            "score": data.score,
            "flaps": 0 
        }).execute()
        
        return {"player": data.player, "difficulty": DEFAULT_DIFFICULTY_INDEX}

    except Exception as e:
        print(f"Supabase Error during adaptive log: {e}")
        return {"player": data.player, "difficulty": DEFAULT_DIFFICULTY_INDEX}


# --- UTILITY ENDPOINTS ---

@app.get("/api/info")
async def get_info():
    """Returns RL parameters and Q-table summary for monitoring."""
    return {
        "status": "RL Server Active",
        "state_space_size": STATES,
        "action_space_size": ACTIONS,
        "q_table_sum": float(np.sum(Q)),
        "q_table_max": float(np.max(Q)),
        "epsilon": EPSILON,
        "gamma": GAMMA,
        "alpha": ALPHA,
    }

@app.get("/api/save_q")
async def save_q_table():
    """Manual endpoint to save the Q-table to disk."""
    try:
        np.save("qtable.npy", Q)
        return {"message": "Q-table saved successfully to qtable.npy"}
    except Exception as e:
        return {"error": f"Failed to save Q-table: {e}"}


if __name__ == '__main__':
    # Running locally uses uvicorn directly
    import uvicorn
    print("Starting FastAPI server locally on http://0.0.0.0:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
