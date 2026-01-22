#!/usr/bin/env python3
"""
Run Bad Reachy - the sarcastic robot assistant.

Usage:
    python run_app.py
"""

import sys
import time


def main():
    from bad_reachy import BadReachyApp
    import os

    print("Starting Bad Reachy... *sigh*")

    # Check if we should use simulation mode
    use_sim = os.getenv("USE_SIM", "false").lower() == "true"

    reachy = None
    if not use_sim:
        from reachy_mini import ReachyMini

        # Retry logic for robust SDK connection
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[REACHY] Connecting to SDK (attempt {attempt+1}/{max_retries})...")

                # Enable daemon auto-start for reliable head movement
                reachy = ReachyMini(spawn_daemon=True, localhost_only=False, timeout=15.0)
                print("[REACHY] Connected to robot hardware!")

                # Check for head control API
                if hasattr(reachy, 'set_target_head_pose'):
                    print("[REACHY] Head control available!")
                else:
                    print("[REACHY] Warning: No head control found")
                break  # Success, exit retry loop

            except Exception as e:
                print(f"[REACHY] Connection attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("[REACHY] SDK connection failed after retries")
                    reachy = None

        if reachy is None:
            print("[REACHY] Running with direct hardware (USB audio/camera)")
    else:
        print("[REACHY] Running in simulation mode (USE_SIM=true)")

    app = BadReachyApp(reachy)
    app.run()


if __name__ == "__main__":
    print("Bad Reachy initializing...")
    print("(Warning: This robot has a bad attitude and swears)")
    print()
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye! Bad Reachy is relieved to stop working.")
        sys.exit(0)
