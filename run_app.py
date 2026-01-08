#!/usr/bin/env python3
"""
Run Grumpy Reachy - the sarcastic robot assistant.

Usage:
    python run_app.py
"""

import sys


def main():
    from reachy_mini import ReachyMini
    from grumpy_reachy import GrumpyReachyApp

    print("Starting Grumpy Reachy... *sigh*")
    reachy = ReachyMini()
    app = GrumpyReachyApp(reachy)
    app.run()


if __name__ == "__main__":
    print("Grumpy Reachy initializing...")
    print("(Warning: This robot has a bad attitude and swears)")
    print()
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye! Grumpy Reachy is relieved to stop working.")
        sys.exit(0)
