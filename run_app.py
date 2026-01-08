#!/usr/bin/env python3
"""
Run Bad Reachy - the sarcastic robot assistant.

Usage:
    python run_app.py
"""

import sys


def main():
    from bad_reachy import BadReachyApp
    import os

    print("Starting Bad Reachy... *sigh*")

    # Check if we should use simulation mode
    use_sim = os.getenv("USE_SIM", "false").lower() == "true"

    reachy = None
    if not use_sim:
        try:
            from reachy_mini import ReachyMini
            # spawn_daemon=True to start our own daemon if needed
            reachy = ReachyMini(spawn_daemon=True, localhost_only=False, timeout=15.0)
            print("[REACHY] Connected to robot hardware")
        except Exception as e:
            print(f"[REACHY] Hardware not available: {e}")
            print("[REACHY] Running in software-only mode (no robot control)")
            reachy = None
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
