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
    reachy = ReachyMini()
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
