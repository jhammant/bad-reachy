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
            import concurrent.futures
            from reachy_mini import ReachyMini

            def connect_reachy():
                """Connect in a thread so we can timeout properly."""
                import sys
                print("[REACHY] Thread starting connection...", flush=True)
                sys.stdout.flush()
                # Try new API first, fall back to old API
                try:
                    from reachy_mini import ConnectionMode
                    print("[REACHY] Using ConnectionMode.LOCAL", flush=True)
                    r = ReachyMini(spawn_daemon=False, connection_mode=ConnectionMode.LOCAL, timeout=30.0)
                    print("[REACHY] SDK returned!", flush=True)
                    return r
                except (ImportError, TypeError) as e:
                    # Old API fallback
                    print(f"[REACHY] Falling back to old API: {e}", flush=True)
                    r = ReachyMini(spawn_daemon=False, localhost_only=False, timeout=30.0)
                    print("[REACHY] SDK returned (old API)!", flush=True)
                    return r

            print("[REACHY] Connecting to existing daemon (45s timeout)...")

            # Use ThreadPoolExecutor for proper timeout handling
            # Note: GStreamer camera init can be slow, needs ~25s
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(connect_reachy)
                try:
                    reachy = future.result(timeout=45.0)
                    print("[REACHY] Connected to robot hardware!")

                    # Check for head control API
                    if hasattr(reachy, 'set_target_head_pose'):
                        print("[REACHY] Head control available!")
                    else:
                        print("[REACHY] Warning: No head control found")

                except concurrent.futures.TimeoutError:
                    print("[REACHY] Connection timed out after 45s")
                    reachy = None
                except Exception as e:
                    print(f"[REACHY] Connection failed: {e}")
                    reachy = None

        except Exception as e:
            print(f"[REACHY] Hardware not available: {e}")
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
