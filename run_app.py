#!/usr/bin/env python3
"""
Run Bad Reachy - the sarcastic robot assistant.

Usage:
    python run_app.py
"""

import sys


def main():
    from reachy_mini import ReachyMini
    from bad_reachy import BadReachyApp

    print("Starting Bad Reachy... *sigh*")
    # spawn_daemon=True to start our own daemon if needed
    # localhost_only=False to connect to any available daemon
    reachy = ReachyMini(spawn_daemon=True, localhost_only=False, timeout=15.0)
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
