"""
train.py — PPO self-play training for BattleSnakeNet.

Usage:
    source .venv/bin/activate
    PYTHONPATH=/path/to/BattleSnake_2026 python3 my_nn_snake/core/train.py

Key design (from TrainingNetworkPlan.md):
  - Curriculum: survival basics → food pressure → combat → full multiplayer
  - Mixed opponent pool: old checkpoints + heuristic bots
  - PPO with value + policy + auxiliary heads
  - Hard replay emphasis on tactical states
  - Board symmetry augmentation (flip + rotate)
"""

import os
import sys
import time
import random
import math
import signal
from pathlib import Path
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from my_nn_snake.core.env           import BattleSnakeEnv, MOVE_NAMES, MOVE_IDX
from my_nn_snake.core.state_encoder import StateEncoder
from my_nn_snake.core.model         import BattleSnakeNet
from my_nn_snake.core.heuristic_filter import get_safe_moves, apply_mask

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR  = Path(__file__).parent / "weights"
BEST_PATH    = WEIGHTS_DIR / "best.pt"
CKPT_PATH    = WEIGHTS_DIR / "latest.pt"

# PPO
LEARNING_RATE   = 3e-4         # Increased for faster sprint learning
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.01
AUX_COEF        = 0.1          # auxiliary heads weight
MAX_GRAD_NORM   = 0.5
PPO_EPOCHS      = 4
MINIBATCH_SIZE  = 1024         # Larger batch for A100s

# Training scale
NUM_ENVS        = 512          # Maximizing A100 VRAM for faster rollout
HISTORY_LEN     = 8
MAX_SNAKES      = 4
ROLLOUT_LEN     = 256
TOTAL_STEPS     = 20_000_000  # Updated to fit an 8-hour window
LOG_INTERVAL    = 1            # Print status every single rollout
SAVE_INTERVAL   = 20           # Save every 20 rollouts
RENDER_INTERVAL = 0
VERBOSE         = False        # verbose rollout printing

# Curriculum stages (steps where we advance)
CURRICULUM = [
    # (step_threshold, num_snakes, board_w, board_h, royale_start)
    (0,          4, 11, 11, 0),     # FORCE Stage C: full multiplayer immediately
    (3_000_000,  4, 11, 11, 1000),  # Royale endgame
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_curriculum_stage(step):
    stage = CURRICULUM[0]
    for s in CURRICULUM:
        if step >= s[0]:
            stage = s
    return stage

def encode_state(data, encoder, history_buf, device="cpu"):
    """Build model inputs (Numpy -> Torch CPU)."""
    history_buf.append(data)
    frames = list(history_buf)
    while len(frames) < HISTORY_LEN:
        frames.insert(0, frames[0])

    boards   = np.stack([encoder.encode_board(f)           for f in frames])   # (8,16,H,W)
    entities = np.stack([encoder.encode_entity_stream(f)   for f in frames])   # (8,8,8)
    scalar   = encoder.get_scalar_context(data)                                 # (6,)

    head_pos = np.array([data["you"]["head"]["x"], data["you"]["head"]["y"]], dtype=np.float32)

    # Return as CPU tensors
    return (
        torch.from_numpy(boards).unsqueeze(0),
        torch.from_numpy(entities).unsqueeze(0),
        torch.from_numpy(scalar).unsqueeze(0),
        torch.from_numpy(head_pos).unsqueeze(0)
    )

def augment(board_seq):
    """Random 90° rotation and flip for data augmentation."""
    k = random.randint(0, 3)
    board_seq = torch.rot90(board_seq, k, dims=[-2, -1])
    if random.random() < 0.5:
        board_seq = torch.flip(board_seq, dims=[-1])
    return board_seq

def run_demo_episode(model, encoder, env_class, stage_cfg, device):
    _, n_snakes, bw, bh, royale_start = stage_cfg
    env = env_class(
        width=bw, height=bh,
        num_snakes=n_snakes,
        royale_shrink_every=25 if royale_start else 0,
        royale_start_turn=royale_start,
        max_turns=1500,
    )
    obs = env.reset()
    hist = {sid: deque(maxlen=HISTORY_LEN) for sid in obs}
    
    print(f"\n{'='*40}")
    print(f" DEMO EPISODE (Stage: {stage_cfg})")
    print(f"{'='*40}\n")
    model.eval()
    
    while True:
        print(env.render())
        print()
        
        actions = {}
        for sid, data in obs.items():
            b, e, s, h = encode_state(data, encoder, hist[sid], device)
            with torch.no_grad():
                policy_logits, *_ = model(b.to(device), e.to(device), s.to(device), h.to(device))
                
            safe = get_safe_moves(data)
            raw_logits = policy_logits[0].cpu().tolist()
            masked_logits = apply_mask(raw_logits, safe, MOVE_IDX)
            masked_t = torch.tensor(masked_logits, dtype=torch.float32)
            
            action = torch.argmax(masked_t).item()
            actions[sid] = MOVE_NAMES[action]
            
        obs, rewards, done, info = env.step(actions)
        if done:
            print(env.render())
            print(f"\n--- DEMO ENDED AT TURN {env.turn} ---")
            for sid, r in rewards.items():
                print(f"{sid} final reward: {r:.2f}")
            print(f"{'='*40}\n")
            break
        
        # Keep training logs from scrolling instantly
        time.sleep(0.1)

class RunningStats:
    """Welford's online mean/std for reward normalisation."""
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2  += delta * (x - self.mean)
    @property
    def std(self): return math.sqrt(self.M2 / max(1, self.n - 1))

# ──────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.board_seqs   = []
        self.entity_seqs  = []
        self.scalars      = []
        self.head_poss    = []
        self.actions      = []
        self.log_probs    = []
        self.values       = []
        self.rewards      = []
        self.dones        = []

    def push(self, b, e, s, h, action, log_p, value, reward, done):
        """Store as CPU tensors to save VRAM."""
        self.board_seqs.append(b.cpu())
        self.entity_seqs.append(e.cpu())
        self.scalars.append(s.cpu())
        self.head_poss.append(h.cpu())
        self.actions.append(action)
        self.log_probs.append(log_p)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_tensors(self):
        return (
            torch.cat(self.board_seqs, dim=0),
            torch.cat(self.entity_seqs, dim=0),
            torch.cat(self.scalars, dim=0),
            torch.cat(self.head_poss, dim=0),
            torch.tensor(self.actions),
            torch.tensor(self.log_probs),
            torch.tensor(self.values),
            torch.tensor(self.rewards),
            torch.tensor(self.dones)
        )
    def __len__(self):
        return len(self.actions)

def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalised Advantage Estimation."""
    if VERBOSE: print(f"  [Verbose] Calculating GAE/Advantages for {len(rewards)} samples...")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae   = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = last_value
        else:
            next_val = values[t + 1]
        delta      = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae   = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns

# ──────────────────────────────────────────────────────────────────────────────
# PPO update
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(model, optimizer, buffer, advantages, returns):
    n = len(buffer)
    indices = list(range(n))

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0

    if VERBOSE: print(f"  [Verbose] Optimizing Policy (PPO Update) | Epochs: {PPO_EPOCHS} | Minibatch: {MINIBATCH_SIZE}")

    for epoch in range(PPO_EPOCHS):
        if VERBOSE: print(f"    [Verbose] Epoch {epoch+1}/{PPO_EPOCHS}...")
        random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb_idx = indices[start : start + MINIBATCH_SIZE]
            if not mb_idx:
                continue

            # Stack minibatch tensors and MOVE TO DEVICE
            device_to_use = next(model.parameters()).device
            mb_b = torch.cat([buffer.board_seqs[i]  for i in mb_idx], dim=0).to(device_to_use)
            mb_e = torch.cat([buffer.entity_seqs[i] for i in mb_idx], dim=0).to(device_to_use)
            mb_s = torch.cat([buffer.scalars[i]     for i in mb_idx], dim=0).to(device_to_use)
            mb_h = torch.cat([buffer.head_poss[i]   for i in mb_idx], dim=0).to(device_to_use)

            # Augmentation on the fly
            mb_b = augment(mb_b)

            mb_actions   = torch.tensor([buffer.actions[i]   for i in mb_idx], dtype=torch.long).to(device_to_use)
            mb_old_logp  = torch.tensor([buffer.log_probs[i] for i in mb_idx], dtype=torch.float32).to(device_to_use)
            mb_adv       = torch.tensor([advantages[i]        for i in mb_idx], dtype=torch.float32).to(device_to_use)
            mb_ret       = torch.tensor([returns[i]           for i in mb_idx], dtype=torch.float32).to(device_to_use)

            # Normalise advantages within the minibatch
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # Normalise returns to stabilise value loss during early training
            mb_ret = (mb_ret - mb_ret.mean()) / (mb_ret.std() + 1e-8)

            # Forward
            policy_logits, value_pred, food_urg, kill_opp, territory = model(mb_b, mb_e, mb_s, mb_h)
            p_dist   = torch.distributions.Categorical(logits=policy_logits)
            new_logp = p_dist.log_prob(mb_actions)
            entropy  = p_dist.entropy().mean()

            # PPO clipped policy loss
            ratio       = torch.exp(new_logp - mb_old_logp)
            surr1       = ratio * mb_adv
            surr2       = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred  = value_pred.squeeze(-1)
            value_loss  = F.mse_loss(value_pred, mb_ret)

            # Auxiliary losses (auxiliary heads produce crude self-supervised signals)
            # food_urgency: snake should predict how hungry it is (health proxy)
            # kill_opp: whether smaller snake is adjacent (rough)
            # territory: fraction of board reachable (rough)
            # We'll just regularise them toward zero for now; real labels come during
            # supervised pretraining phase. They still improve gradient flow.
            aux_loss = (food_urg.mean() ** 2 + kill_opp.mean() ** 2 + territory.mean() ** 2)

            loss = (policy_loss
                    + VALUE_COEF  * value_loss
                    - ENTROPY_COEF * entropy
                    + AUX_COEF    * aux_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()

    batches = PPO_EPOCHS * max(1, n // MINIBATCH_SIZE)
    return (total_policy_loss / batches,
            total_value_loss  / batches,
            total_entropy     / batches)

# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO Training for BattleSnake")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--render-interval", type=int, default=RENDER_INTERVAL)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--verbose", action="store_true", default=VERBOSE)
    parser.add_argument("--load", type=str, default=None, help="latest, best, or path")
    args = parser.parse_args()

    # --- DDP INITIALIZATION ---
    dist_url = "env://"
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = DEVICE

    is_master = rank == 0

    if is_master:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Training on {device} (Distributed: {is_distributed}, World Size: {world_size})", flush=True)

    encoder = StateEncoder(history_length=HISTORY_LEN)
    model   = BattleSnakeNet().to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Determine which file to load
    target_load = None
    if args.load:
        if args.load == "best":   target_load = BEST_PATH
        elif args.load == "latest": target_load = CKPT_PATH
        else: target_load = Path(args.load)
    elif CKPT_PATH.exists():
        target_load = CKPT_PATH

    start_step = 0
    if target_load and target_load.exists():
        ckpt = torch.load(target_load, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            raw_model.load_state_dict(ckpt["model"])
            if "optim" in ckpt: optimizer.load_state_dict(ckpt["optim"])
            start_step = ckpt.get("step", 0)
        else:
            raw_model.load_state_dict(ckpt)
        if is_master:
            print(f"Loaded weights from {target_load} (start_step={start_step})")

    # Graceful Ctrl+C handling
    interrupted = [False]
    def _sig(sig, frame):
        interrupted[0] = True
        if is_master:
            print("\nInterrupted – finishing current update then saving...")
    signal.signal(signal.SIGINT, _sig)

    # Running stats for reward normalisation
    reward_stats = RunningStats()
    best_avg_reward = -float("inf")

    global_step   = start_step
    update_count  = 0
    ep_rewards    = deque(maxlen=100)   # episode reward tracker
    ep_lengths    = deque(maxlen=100)

    # Build initial environments
    stage_cfg = _get_curriculum_stage(global_step)
    _, n_snakes, bw, bh, royale_start = stage_cfg
    envs = [
        BattleSnakeEnv(
            width=bw, height=bh,
            num_snakes=n_snakes,
            royale_shrink_every=25 if royale_start else 0,
            royale_start_turn=royale_start,
            max_turns=1500,
        )
        for _ in range(NUM_ENVS)
    ]
    obs_list   = [env.reset() for env in envs]
    histories  = [{sid: deque(maxlen=HISTORY_LEN) for sid in obs}
                  for obs in obs_list]
    ep_rew     = [{sid: 0.0 for sid in obs} for obs in obs_list]

    t0 = time.time()
    if is_master:
        print(f"Starting PPO training  |  total_steps={args.total_steps:,}  |  envs={NUM_ENVS} per GPU")
        print(f"Stage: {stage_cfg}")

    while global_step < args.total_steps and not interrupted[0]:
        # --- Update model to eval mode ---
        model.eval()
        
        # --- (Rest of the training loop stays roughly the same, but logging only on is_master) ---
        # NOTE: Make sure to wrap logging prints in 'if is_master:'


        # ── Rebuild environments if curriculum stage changed ──
        new_stage = _get_curriculum_stage(global_step)
        if new_stage != stage_cfg:
            stage_cfg = new_stage
            _, n_snakes, bw, bh, royale_start = stage_cfg
            if is_master:
                print(f"\n[step {global_step:,}] Curriculum advance → {stage_cfg}")
            envs = [
                BattleSnakeEnv(
                    width=bw, height=bh,
                    num_snakes=n_snakes,
                    royale_shrink_every=25 if royale_start else 0,
                    royale_start_turn=royale_start,
                    max_turns=1500,
                )
                for _ in range(NUM_ENVS)
            ]
            obs_list   = [env.reset() for env in envs]
            histories  = [{sid: deque(maxlen=HISTORY_LEN) for sid in obs}
                          for obs in obs_list]
            ep_rew     = [{sid: 0.0 for sid in obs} for obs in obs_list]

        # ── Collect rollout ──
        buffer = RolloutBuffer()
        model.eval()

        for step_i in range(ROLLOUT_LEN):
            if is_master and VERBOSE and step_i % max(1, ROLLOUT_LEN // 10) == 0:
                print(f"  [Verbose] Rollout Step {step_i}/{ROLLOUT_LEN}...")

            # BATCHED ENCODING (CPU)
            b_list, e_list, s_list, h_list = [], [], [], []
            for i in range(NUM_ENVS):
                for sid, data in obs_list[i].items():
                    b, e, s, h = encode_state(data, encoder, histories[i][sid], "cpu")
                    b_list.append(b); e_list.append(e); s_list.append(s); h_list.append(h)
            
            # MOVE ENTIRE BATCH TO GPU AT ONCE
            batch_b = torch.cat(b_list, dim=0).to(device)
            batch_e = torch.cat(e_list, dim=0).to(device)
            batch_s = torch.cat(s_list, dim=0).to(device)
            batch_h = torch.cat(h_list, dim=0).to(device)
            
            with torch.no_grad():
                policy_logits, value_preds, *_ = model(batch_b, batch_e, batch_s, batch_h)
            
            # Map back to envs and step
            idx = 0
            for i in range(NUM_ENVS):
                env_actions = {}
                # In current stage, there might be multiple snakes per env
                # but we usually train one snake at a time or vs bots.
                # However, the code supports multi-agent envs.
                env_sids = list(obs_list[i].keys())
                for sid in env_sids:
                    data = obs_list[i][sid]
                    safe = get_safe_moves(data)
                    p_logits = policy_logits[idx]
                    
                    masked_logits = apply_mask(p_logits.cpu().tolist(), safe, MOVE_IDX)
                    masked_t = torch.tensor(masked_logits, dtype=torch.float32)
                    p_dist = torch.distributions.Categorical(logits=masked_t)
                    
                    action = p_dist.sample()
                    log_p  = p_dist.log_prob(action).item()
                    val    = value_preds[idx].item()
                    
                    env_actions[sid] = MOVE_NAMES[action.item()]
                    
                    # Store rollout data (using the tensors we already created)
                    # We store them frame-by-frame to the buffer
                    buffer.push(b_list[idx], e_list[idx], s_list[idx], h_list[idx], 
                                action.item(), log_p, val, 0, False) # rewards/dones updated after step
                    idx += 1
                
                # Step the environment
                new_obs, rewards, done, info = envs[i].step(env_actions)
                
                # Update rewards and dones for previous buffer push
                # (A bit tricky since buffer is global, we need indices)
                # For simplicity, we store the reward of the action that led to this state.
                # Since buffer is just a list, we go back and fix the last 'num_snakes' entries
                for j in range(len(env_sids)):
                    back_idx = -(len(env_sids) - j)
                    sid = env_sids[j]
                    buffer.rewards[back_idx] = float(rewards.get(sid, 0.0))
                    buffer.dones[back_idx]   = float(done)

                    if done:
                        # Episode finished
                        ep_rewards.append(ep_rew[i][sid] + rewards.get(sid, 0.0))
                        ep_lengths.append(info["turn"])
                        ep_rew[i][sid] = 0.0
                        # Env reset handled below
                    else:
                        ep_rew[i][sid] += rewards.get(sid, 0.0)

                if done:
                    obs_list[i] = envs[i].reset()
                    histories[i] = {sid: deque(maxlen=HISTORY_LEN) for sid in obs_list[i]}
                    ep_rew[i] = {sid: 0.0 for sid in obs_list[i]}
                else:
                    obs_list[i] = new_obs

            global_step += NUM_ENVS * world_size # Total steps across all GPUs

        # ── Bootstrap last value ──
        last_vals = []
        for sid, data in list(obs_list[0].items())[:1]:
            b, e, s, h = encode_state(data, encoder, deque(maxlen=HISTORY_LEN), device)
            with torch.no_grad():
                _, val, *_ = model(b.to(device), e.to(device), s.to(device), h.to(device))
            last_vals.append(val.item())
        last_value = last_vals[0] if last_vals else 0.0

        # ── GAE ──
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_value
        )

        # ── PPO update ──
        model.train()
        p_loss, v_loss, ent = ppo_update(model, optimizer, buffer, advantages, returns)
        update_count += 1

        # ── Logging ──
        if update_count % LOG_INTERVAL == 0:
            # Global rolling metrics
            avg_rew = np.mean(ep_rewards)   if ep_rewards   else 0.0
            avg_len = np.mean(ep_lengths)   if ep_lengths   else 0.0
            elapsed = time.time() - t0
            sps     = global_step / elapsed
            if is_master:
                print(
                    f"step={global_step:>8,}  "
                    f"upd={update_count:>5}  "
                    f"rew={avg_rew:>7.3f}  "
                    f"len={avg_len:>6.1f}  "
                    f"p_loss={p_loss:.4f}  "
                    f"v_loss={v_loss:.4f}  "
                    f"ent={ent:.4f}  "
                    f"sps={sps:,.0f}",
                    flush=True
                )

            if is_master and avg_rew > best_avg_reward and len(ep_rewards) >= 10:
                best_avg_reward = avg_rew
                torch.save(raw_model.state_dict(), BEST_PATH)
                print(f"  ✅ New best model saved  (avg_rew={avg_rew:.4f})")

        # ── Demo ──
        if is_master and RENDER_INTERVAL > 0 and update_count % RENDER_INTERVAL == 0:
            run_demo_episode(raw_model, encoder, BattleSnakeEnv, stage_cfg, device)

        # ── Checkpoint ──
        if is_master and update_count % SAVE_INTERVAL == 0:
            torch.save({"model": raw_model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step":  global_step}, CKPT_PATH)
            print(f"  💾 Checkpoint saved @ step {global_step:,}")

    if is_master:
        torch.save({"model": raw_model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step":  global_step}, CKPT_PATH)
        torch.save(raw_model.state_dict(), BEST_PATH)
        print(f"\nTraining complete. Steps: {global_step:,}. Weights saved to {WEIGHTS_DIR}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
