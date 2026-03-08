# convLSTM-battlesnake: Multi-Node Deep RL Battlesnake AI

![Battlesnake](https://img.shields.io/badge/Battlesnake-2026-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)

convLSTM-battlesnake is a deep reinforcement learning Battlesnake AI designed for the 2026 NJIT Battlesnakes Tournament. It features a custom neural architecture trained using PPO (Proximal Policy Optimization) on multi-node A100 clusters.

## 🚀 Key Features
- **Temporal Memory**: A ConvLSTM backbone that processes turn history to predict opponent paths.
- **Distributed Training**: Fully DDP-compatible training script, scaled across 8x NVIDIA A100 GPUs.
- **Safety First**: Integrated heuristic filtering (Flood Fill, Collision Detection) as a hard mask over neural policy.
- **Multi-Modal Fusion**: Combines spatial board data, individual snake embeddings, and scalar game context.

## 📁 Repository Structure
- `my_nn_snake/`: Core logic and model definition.
  - `core/model.py`: The ConvLSTM + Self-Attention architecture.
  - `core/train.py`: The Distributed Reinforcement Learning (PPO) pipeline.
  - `core/env.py`: High-performance Battlesnake environment simulator.
  - `core/state_encoder.py`: Board-to-tensor encoding logic.
  - `core/heuristic_filter.py`: Python-based safety heuristics.
- `play_model.py`: Main entry point to launch the bot for matches.
- `cluster/`: SLURM scripts for supercomputer deployment.
- `Dockerfile`: Containerized deployment support.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support for training)
- Battlesnake CLI

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/giovannettif/convLSTM-battlesnake.git
   cd convLSTM-battlesnake
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the bot:
   ```bash
   PYTHONPATH=. python3 -m uvicorn my_nn_snake.main:app --port 8000
   ```

## 🧠 Training Info
The model was trained on an 11x11 grid with 4 snakes using a curriculum-based PPO approach. The final tournament sprint focused on Stage C high-intensity combat.

## 🌟 About convLSTM-battlesnake

### 💡 Inspiration
The inspiration for **convLSTM-battlesnake** came from the realization that standard Battlesnake bots often play too "instinctively," making decisions based only on the current frame. We wanted to build a snake with a **temporal memory**—one that could "see" the history of its opponents' trails to predict where they are heading, not just where they are.

### 🐍 What it does
It is a Deep Reinforcement Learning agent that competes in 4-snake multiplayer matches. It doesn't just avoid walls; it learns high-level tactical behaviors like "coiling" to protect territory, "cutoff" maneuvers to trap opponents, and health-based aggression. It uses a safety-first heuristic layer to ensure it never makes a basic mistake like hitting a wall while the neural network handles the complex strategy.

### 🛠️ How we built it
- **Framework**: Built with **PyTorch** and **FastAPI**.
- **Model**: A custom architecture featuring a **ConvLSTM backbone** for temporal feature extraction and **Multi-Head Self-Attention** for cross-snake interaction.
- **Compute**: Optimized for the **RTX 4060** locally, but scaled to **8x NVIDIA A100 GPUs** on the Wulver Supercomputer cluster.
- **Algorithm**: Used **Proximal Policy Optimization (PPO)** with a custom curriculum designed for high-density combat.

### 🚧 Challenges we ran into
- **Distributed Scaling**: We faced significant hurdles with **Distributed Data Parallel (DDP)** orchestration, specifically solving rendezvous timeouts between compute nodes and the login node.
- **Out-of-Bounds Edge Cases**: We spent critical time debugging a rare `IndexError` that only appeared during high-intensity 4-snake collisions on the cluster. We solved this with robust state clipping in the model's head-feature extraction.
- **The Clock**: Designing and training a 500-million-step model with matches only an hour away forced us to pivot to a **Tournament Sprint** curriculum, jumping straight into combat training.

### 🏆 Accomplishments that we're proud of
- **8-GPU Parallelism**: Successfully scaling training across 2 nodes (8 GPUs) to reach a throughput of over **4 million steps per minute**.
- **Stability**: Fixing a critical crash-on-death bug 15 minutes before the tournament deadline.
- **Unified Pipeline**: Creating a standardized setup that works seamlessly across local hardware and supercomputing clusters.

### 📖 What we learned
We learned the intricacies of **multi-node distributed training**, the importance of differentiable state encoding, and how to effectively "force-curriculum" a model when time is the primary constraint. We also gained deep experience in debugging remote SLURM environments.

### 🚀 What's next for convLSTM
- **Royale Mode Mastery**: Further tuning for the 1000+ turn "Royale" endgame where the board shrinks.
- **Territory Heuristics**: Deepening the reward function to prioritize area control even when enemies are far away.
- **Long-Term Evolution**: Running the full 500-million-step curriculum over multiple days to see the limits of its strategic depth.

## ⚖️ License
MIT License. See [LICENSE](LICENSE) for details.
