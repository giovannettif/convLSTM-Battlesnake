import subprocess
import time
import os
import sys
import argparse
import socket
from pathlib import Path

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def play_model(weights="latest", port=8001, browser=True, venv=None, num_snakes=1):
    """
    Launches a Battlesnake bot server using the trained model weights
    and then starts a game against it using the CLI.
    """
    root = Path(__file__).parent.absolute()
    
    # 1. Determine weights path
    weights_map = {
        "latest": "my_nn_snake/core/weights/latest.pt",
        "best": "my_nn_snake/core/weights/best.pt"
    }
    w_path = weights_map.get(weights, weights)
    if not os.path.isabs(w_path):
        w_path = str(root / w_path)
    
    if not os.path.exists(w_path):
        print(f"❌ Error: Weights file not found: {w_path}")
        return

    print(f"🐍 Loading model from: {w_path}")
    
    # 2. Start the Bot Server in the background
    print(f"🚀 Starting Bot Server on port {port}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["SNAKE_WEIGHTS"] = w_path
    
    # Determine which python to use for uvicorn
    py_exec = sys.executable
    if venv:
        v_py = Path(venv) / "bin" / "python3"
        if v_py.exists():
            py_exec = str(v_py)
            print(f"📂 Using VENV Python: {py_exec}")
    elif Path(".venv").exists():
        v_py = root / ".venv" / "bin" / "python3"
        if v_py.exists():
            py_exec = str(v_py)

    bot_proc = subprocess.Popen(
        [py_exec, "-m", "uvicorn", "my_nn_snake.main:app", "--port", str(port), "--host", "127.0.0.1", "--log-level", "error"],
        env=env
    )
    
    # Wait for the server to be ready
    print(f"⏳ Waiting for server to initialize (up to 15s)...")
    ready = False
    for _ in range(30): # 15 seconds max (0.5s intervals)
        if is_port_open(port):
            ready = True
            break
        if bot_proc.poll() is not None:
            print("❌ Error: Bot server process exited prematurely.")
            return
        time.sleep(0.5)
    
    if not ready:
        print(f"❌ Error: Bot server failed to respond on port {port}.")
        bot_proc.terminate()
        return

    print("✅ Server is ready!")
    
    # 3. Launch the Battlesnake Game using the CLI
    print(f"🎮 Launching Game Engine with {num_snakes} snakes...")
    cli_cmd = ["./battlesnake", "play"]
    for i in range(num_snakes):
        cli_cmd += ["--url", f"http://127.0.0.1:{port}", "--name", f"Snake_{i+1}"]
        
    cli_cmd += [
        "--width", "11", "--height", "11",
        "--delay", "200",          # Fast but watchable
        "--minimumFood", "1",
        "--foodSpawnChance", "15",
    ]
    
    if browser:
        cli_cmd += ["--browser", "--board-url", "http://localhost:3000"]
    
    try:
        subprocess.run(cli_cmd)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bot_proc.terminate()
        bot_proc.wait()
        print("✅ Cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a game with the trained model.")
    parser.add_argument("--weights", type=str, default="latest", help="Choose 'latest', 'best', or a direct path.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the bot server on.")
    parser.add_argument("--num-snakes", type=int, default=1, help="Number of instances of the model to play against each other.")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically.")
    parser.add_argument("--venv", type=str, default=None, help="Path to virtual environment.")
    args = parser.parse_args()
    
    play_model(weights=args.weights, port=args.port, browser=not args.no_browser, venv=args.venv, num_snakes=args.num_snakes)
