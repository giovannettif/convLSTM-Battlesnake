# convLSTM-battlesnake: Multi-Node Deep RL Battlesnake AI

![Battlesnake](https://img.shields.io/badge/Battlesnake-2026-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)

convLSTM-battlesnake is a deep reinforcement learning Battlesnake AI designed for the 2026 NJIT Battlesnakes Winter Tournament. It features a custom neural architecture trained using PPO (Proximal Policy Optimization) on multi-node A100 clusters.

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

## ⚖️ License
MIT License. See [LICENSE](LICENSE) for details.
