"""
Head Wobbler - Synchronizes head movement with audio playback.
Based on Pollen Robotics reference implementation.

Runs in a background thread, processing audio data and applying
natural head movements synchronized to speech.
"""

from __future__ import annotations
import time
import queue
import base64
import threading
from typing import Tuple, Callable, Optional
from collections.abc import Callable as CallableABC

import numpy as np
from numpy.typing import NDArray

from .speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 16000  # Default sample rate
MOVEMENT_LATENCY_S = 0.08  # Seconds between audio and robot movement


class HeadWobbler:
    """
    Converts audio data into synchronized head movements.

    Feed audio bytes/arrays and the wobbler will apply natural
    head movements to the robot synchronized with speech.
    """

    def __init__(
        self,
        set_offsets_callback: Callable[[Tuple[float, float, float]], None],
        sample_rate: int = SAMPLE_RATE
    ):
        """
        Initialize the head wobbler.

        Args:
            set_offsets_callback: Function to call with (pitch, yaw, roll) in radians
            sample_rate: Audio sample rate (default 16000)
        """
        self._apply_offsets = set_offsets_callback
        self._sample_rate = sample_rate
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16]]]" = queue.Queue()
        self.sway = SwayRollRT()

        # Synchronization
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    def feed_bytes(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> None:
        """
        Feed raw audio bytes (int16 PCM) into the wobbler.

        Args:
            audio_bytes: Raw PCM audio data (int16)
            sample_rate: Sample rate of the audio (uses default if None)
        """
        sr = sample_rate or self._sample_rate
        try:
            buf = np.frombuffer(audio_bytes, dtype=np.int16)
            with self._state_lock:
                generation = self._generation
            self.audio_queue.put((generation, sr, buf))
        except Exception as e:
            print(f"[HeadWobbler] Error feeding bytes: {e}")

    def feed_array(self, audio_array: NDArray, sample_rate: Optional[int] = None) -> None:
        """
        Feed numpy array audio data into the wobbler.

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of the audio (uses default if None)
        """
        sr = sample_rate or self._sample_rate
        try:
            if audio_array.dtype == np.float32:
                # Convert float32 [-1,1] to int16
                buf = (audio_array * 32767).astype(np.int16)
            else:
                buf = audio_array.astype(np.int16)

            with self._state_lock:
                generation = self._generation
            self.audio_queue.put((generation, sr, buf))
        except Exception as e:
            print(f"[HeadWobbler] Error feeding array: {e}")

    def feed_base64(self, delta_b64: str, sample_rate: Optional[int] = None) -> None:
        """
        Feed base64-encoded audio data into the wobbler.

        Args:
            delta_b64: Base64 encoded PCM audio
            sample_rate: Sample rate of the audio
        """
        sr = sample_rate or self._sample_rate
        try:
            buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16)
            with self._state_lock:
                generation = self._generation
            self.audio_queue.put((generation, sr, buf))
        except Exception as e:
            print(f"[HeadWobbler] Error feeding base64: {e}")

    def start(self) -> None:
        """Start the head wobbler background thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._working_loop, daemon=True, name="HeadWobbler")
        self._thread.start()
        print("[HeadWobbler] Started")

    def stop(self) -> None:
        """Stop the head wobbler background thread."""
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[HeadWobbler] Stopped")

    def reset(self) -> None:
        """Reset internal state (call when speech is interrupted)."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0

        # Drain queued audio
        drained = 0
        while True:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                drained += 1
            except queue.Empty:
                break

        with self._sway_lock:
            self.sway.reset()

        if drained > 0:
            print(f"[HeadWobbler] Reset - drained {drained} queued chunks")

    def _working_loop(self) -> None:
        """Background thread that processes audio and applies movements."""
        hop_dt = HOP_MS / 1000.0

        while not self._stop_event.is_set():
            try:
                chunk_generation, sr, chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Check if this chunk is from current generation
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                # Initialize timestamp on first chunk
                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                # Process audio through sway analyzer
                pcm = np.asarray(chunk).flatten()
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                # Apply movements with proper timing
                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    # Calculate target time for this hop
                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    # Handle lag - skip hops if we're behind
                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                            i += drop
                            continue

                    # Wait until target time
                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    # Apply the movement
                    r = results[i]
                    try:
                        self._apply_offsets((
                            r["pitch_rad"],
                            r["yaw_rad"],
                            r["roll_rad"]
                        ))
                    except Exception as e:
                        print(f"[HeadWobbler] Error applying offsets: {e}")

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1

            finally:
                self.audio_queue.task_done()

    @property
    def is_running(self) -> bool:
        """Check if the wobbler is running."""
        return self._running
