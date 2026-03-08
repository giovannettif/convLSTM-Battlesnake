import subprocess
import sys
import argparse
import time

def play_game(headless=False, quiet=False):
    if headless:
        if not quiet: print("🚀 Starting Battlesnake Game Headlessly (Instant)...")
        replay_file = f"replay_{int(time.time())}.json"
        cmd = [
            "./battlesnake", "play",
            "--url", "http://127.0.0.1:8000",
            "--name", "MyNeuralSnake",
            "--output", replay_file
        ]
    else:
        print("🚀 Starting Battlesnake Game Natively...")
        # Run the local binary, connecting local port 8000 (snake) to local port 8080 (engine automatically chosen by CLI)
        cmd = [
            "./battlesnake", "play",
            "--url", "http://127.0.0.1:8000",
            "--name", "MyNeuralSnake",
            "--delay", "500", # Slower to watch
            "--browser",
            "--board-url", "http://localhost:3000"
        ]
    
    try:
        # If quiet, we ignore output. Otherwise, we stream it.
        stdout_dest = subprocess.DEVNULL if quiet else subprocess.PIPE
        stderr_dest = subprocess.STDOUT if not quiet else subprocess.DEVNULL
        
        proc = subprocess.Popen(cmd, stdout=stdout_dest, stderr=stderr_dest, text=True)
        
        if not quiet:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                    
        proc.wait()
        if headless and not quiet:
            print(f"\n✅ Game finished instantly. Replay saved to: {replay_file}")
            print("You can view it by uploading the JSON file to: https://play.battlesnake.com/replay/")
        elif headless and quiet:
             print(f"✅ Game finished. Replay: {replay_file}")
        
    except KeyboardInterrupt:
        if not quiet: print("\nStopping game...")
        proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Battlesnake game.")
    parser.add_argument("--headless", action="store_true", help="Run the game instantly without opening the browser and save a replay JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress all CLI output (logs) and only print the final status.")
    parser.add_argument("--count", type=int, default=1, help="Number of games to run (for bulk testing).")
    args = parser.parse_args()
    
    for i in range(args.count):
        if args.count > 1 and not args.quiet:
            print(f"\n--- Running Game {i+1}/{args.count} ---")
        play_game(headless=args.headless, quiet=args.quiet)
