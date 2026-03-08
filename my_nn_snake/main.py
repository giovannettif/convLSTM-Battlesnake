"""
main.py — Battlesnake bot using BattleSnakeNet + heuristic_filter safety mask.

Architecture:
  /move  →  StateEncoder  →  BattleSnakeNet  →  heuristic_filter (mask)  →  argmax
"""

import os
import random
import sys
from collections import deque
from pathlib import Path

# Allow imports from anywhere in the project when running from BattleSnake_2026/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from fastapi import FastAPI, Request

from my_nn_snake.core.state_encoder import StateEncoder
from my_nn_snake.core.model import BattleSnakeNet
from my_nn_snake.core.heuristic_filter import get_safe_moves, apply_mask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HISTORY_LEN   = 8          # frames of temporal context fed to ConvLSTM
WEIGHTS_PATH  = Path(__file__).parent / "core" / "weights" / "best.pt"
MOVE_IDX      = {"up": 0, "down": 1, "left": 2, "right": 3}
IDX_MOVE      = {v: k for k, v in MOVE_IDX.items()}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model & Encoder — loaded once at startup
# ---------------------------------------------------------------------------
_encoder = StateEncoder(history_length=HISTORY_LEN)
_model   = BattleSnakeNet().to(DEVICE)
_model.eval()

# Check env var for weights (play_model.py uses this)
SNAKE_WEIGHTS = os.getenv("SNAKE_WEIGHTS")
target_path = Path(SNAKE_WEIGHTS) if SNAKE_WEIGHTS else WEIGHTS_PATH

if target_path.exists():
    checkpoint = torch.load(target_path, map_location=DEVICE)
    # Handle if it's a full training checkpoint (dict) or just the state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        _model.load_state_dict(checkpoint["model"])
    else:
        _model.load_state_dict(checkpoint)
    print(f"[BattleSnakeNet] Loaded weights from {target_path}")
else:
    print(f"[BattleSnakeNet] No weights found at {target_path} – using random")

# Per-game frame buffers: {game_id: deque of board-state dicts}
_history: dict[str, deque] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/")
def on_info():
    return {
        "apiversion": "1",
        "author":     "convLSTM-battlesnake",
        "color":      "#1A1A2E",
        "head":       "default",
        "tail":       "default",
    }


@app.post("/start")
async def on_start(request: Request):
    data = await request.json()
    game_id = data["game"]["id"]
    _history[game_id] = deque(maxlen=HISTORY_LEN)
    return "ok"


@app.post("/move")
async def on_move(request: Request):
    data = await request.json()
    game_id = data["game"]["id"]

    # ---- 1. Safety mask (pure Python, no model needed) ----
    safe = get_safe_moves(data)
    any_safe = any(safe.values())

    # ---- 2. Update history buffer ----
    if game_id not in _history:
        _history[game_id] = deque(maxlen=HISTORY_LEN)

    buf = _history[game_id]
    buf.append(data)

    # ---- 3. Build input tensors ----
    # Pad early turns by repeating the current frame
    frames = list(buf)
    while len(frames) < HISTORY_LEN:
        frames.insert(0, frames[0])

    board_frames  = [_encoder.encode_board(f)           for f in frames]
    entity_frames = [_encoder.encode_entity_stream(f)   for f in frames]

    board_seq  = torch.tensor(np.stack(board_frames),  dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,8,16,H,W)
    entity_seq = torch.tensor(np.stack(entity_frames), dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,8,8,8)
    scalar_ctx = torch.tensor(_encoder.get_scalar_context(data), dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,6)

    head = data["you"]["head"]
    head_pos = torch.tensor([[head["x"], head["y"]]], dtype=torch.float32).to(DEVICE)  # (1,2)

    # ---- 4. Forward pass ----
    with torch.no_grad():
        policy_logits, *_ = _model(board_seq, entity_seq, scalar_ctx, head_pos)

    logits = policy_logits[0].cpu().tolist()  # [up, down, left, right]

    # ---- 5. Apply hard safety mask ----
    masked_logits = apply_mask(logits, safe, MOVE_IDX)

    # ---- 6. Pick best safe move ----
    if any_safe:
        best_idx  = int(max(range(4), key=lambda i: masked_logits[i]))
        best_move = IDX_MOVE[best_idx]
    else:
        # Absolute last resort – all moves lethal, pick any
        best_move = random.choice(list(MOVE_IDX.keys()))

    return {"move": best_move}


@app.post("/end")
async def on_end(request: Request):
    data = await request.json()
    game_id = data["game"]["id"]
    _history.pop(game_id, None)
    return "ok"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
