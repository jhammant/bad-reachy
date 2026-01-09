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
            print("[REACHY] Connecting to SDK...")

            # Simple connection like dyson-mechanic uses
            reachy = ReachyMini()
            print("[REACHY] Connected to robot hardware!")

            # Check for head control API
            if hasattr(reachy, 'set_target_head_pose'):
                print("[REACHY] Head control available!")
            else:
                print("[REACHY] Warning: No head control found")

        except Exception as e:
            print(f"[REACHY] SDK connection failed: {e}")
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
