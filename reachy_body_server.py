#!/usr/bin/env python3
"""
Reachy Body Server - Runs on Reachy to provide remote audio I/O and robot control.

The Mac runs the brain (STT, LLM, TTS), this server provides:
- Microphone audio streaming to Mac
- Audio playback on Reachy speaker
- Head movement control
- Emotion expressions

Usage:
    python reachy_body_server.py [--port 8092]
"""

import asyncio
import argparse
import io
import time
import wave
import threading
import numpy as np
from typing import Optional
from collections import deque

from fastapi import FastAPI, Response, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Audio
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[AUDIO] sounddevice not available")

# Reachy SDK
try:
    from reachy_mini import ReachyMini
    REACHY_SDK_AVAILABLE = True
except ImportError:
    REACHY_SDK_AVAILABLE = False
    print("[REACHY] SDK not available")

app = FastAPI(title="Reachy Body Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
reachy = None
audio_input_device = None   # reachymini_audio_src (input)
audio_output_device = None  # reachymini_audio_sink (output)
sample_rate = 16000
is_recording = False
audio_buffer = deque(maxlen=sample_rate * 10)  # 10 seconds buffer
current_emotion = "neutral"

# Reachy Mini daemon API URL (running on same machine)
DAEMON_API_URL = "http://localhost:8000"


def init_reachy():
    """Initialize Reachy SDK connection."""
    global reachy

    if not REACHY_SDK_AVAILABLE:
        print("[REACHY] SDK not available, running without robot control")
        return False

    try:
        print("[REACHY] Connecting to daemon...")
        reachy = ReachyMini(spawn_daemon=False, localhost_only=True, timeout=10.0)
        print("[REACHY] Connected!")
        return True
    except Exception as e:
        print(f"[REACHY] Connection failed: {e}")
        reachy = None
        return False


def init_audio():
    """Initialize audio devices."""
    global audio_input_device, audio_output_device

    if not SOUNDDEVICE_AVAILABLE:
        print("[AUDIO] sounddevice not available")
        return False

    try:
        devices = sd.query_devices()
        print("[AUDIO] Available devices:")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev['name']} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")
            if 'reachy' in dev['name'].lower() or 'pollen' in dev['name'].lower():
                if dev['max_input_channels'] > 0:
                    audio_input_device = i
                    print(f"  -> Selected as INPUT device")
                if dev['max_output_channels'] > 0:
                    audio_output_device = i
                    print(f"  -> Selected as OUTPUT device")

        if audio_input_device is not None and audio_output_device is not None:
            sd.default.device = (audio_input_device, audio_output_device)
            sd.default.samplerate = sample_rate
            print(f"[AUDIO] Input: device {audio_input_device}, Output: device {audio_output_device} at {sample_rate}Hz")
            return True
        else:
            # Use default device
            print("[AUDIO] Using default audio device")
            return True

    except Exception as e:
        print(f"[AUDIO] Init failed: {e}")
        return False


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    init_audio()
    init_reachy()


@app.get("/status")
async def status():
    """Get server status."""
    return {
        "status": "ready",
        "reachy_connected": reachy is not None,
        "audio_available": SOUNDDEVICE_AVAILABLE,
        "audio_input_device": audio_input_device,
        "audio_output_device": audio_output_device,
        "sample_rate": sample_rate,
        "current_emotion": current_emotion,
    }


@app.get("/audio/capture")
async def capture_audio(duration: float = 3.0):
    """
    Capture audio from microphone for specified duration.
    Returns WAV audio bytes.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return Response(content=b"", status_code=503)

    try:
        print(f"[AUDIO] Capturing {duration}s...")

        # Record audio from input device
        samples = int(duration * sample_rate)
        audio_data = sd.rec(
            samples,
            samplerate=sample_rate,
            channels=2,  # Reachy Mini Audio is stereo
            dtype='int16',
            device=audio_input_device
        )
        sd.wait()

        # Convert stereo to mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
            audio_mono = np.mean(audio_data, axis=1).astype(np.int16)
        else:
            audio_mono = audio_data.flatten().astype(np.int16)

        # Check audio level
        max_level = np.max(np.abs(audio_mono.astype(np.float32) / 32767.0))
        print(f"[AUDIO] Captured {len(audio_mono)} samples, max level: {max_level:.3f}")

        # Convert to WAV
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_mono.tobytes())

        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"[AUDIO] Capture error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=b"", status_code=500)


@app.post("/audio/play")
async def play_audio(audio: bytes = Form(...)):
    """
    Play audio on Reachy's speaker.
    Accepts WAV or raw PCM audio.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return {"status": "error", "message": "Audio not available"}

    try:
        # Try to parse as WAV first
        try:
            buffer = io.BytesIO(audio)
            with wave.open(buffer, 'rb') as wf:
                audio_sr = wf.getframerate()
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        except:
            # Assume raw PCM int16 at sample_rate
            audio_data = np.frombuffer(audio, dtype=np.int16)
            audio_sr = sample_rate

        # Convert to float32 for playback
        audio_float = audio_data.astype(np.float32) / 32767.0

        print(f"[AUDIO] Playing {len(audio_float)} samples at {audio_sr}Hz")

        # Play audio on output device
        sd.play(audio_float, samplerate=audio_sr, device=audio_output_device)
        sd.wait()

        return {"status": "ok", "samples": len(audio_float), "duration": len(audio_float) / audio_sr}

    except Exception as e:
        print(f"[AUDIO] Play error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/audio/play_file")
async def play_audio_file(
    duration_hint: float = Form(0),
):
    """
    Play audio sent as request body (not form data).
    Use this for binary audio data.
    """
    from starlette.requests import Request
    # This endpoint is for when we send raw binary in body
    return {"status": "error", "message": "Use /audio/play_bytes instead"}


from starlette.requests import Request

@app.post("/audio/play_bytes")
async def play_audio_bytes(request: Request):
    """
    Play raw audio bytes sent in request body.
    Supports WAV and MP3.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return {"status": "error", "message": "Audio not available"}

    try:
        audio_bytes = await request.body()

        if not audio_bytes:
            return {"status": "error", "message": "No audio data"}

        # Check format
        is_mp3 = audio_bytes[:3] == b'ID3' or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0)

        if is_mp3:
            # Decode MP3 using ffmpeg (pydub doesn't work on Python 3.13)
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_mp3.write(audio_bytes)
                tmp_mp3_path = tmp_mp3.name
            tmp_wav_path = tmp_mp3_path.replace('.mp3', '.wav')
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', tmp_mp3_path, '-ar', '16000', '-ac', '1', tmp_wav_path],
                    capture_output=True, check=True
                )
                with wave.open(tmp_wav_path, 'rb') as wf:
                    audio_sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            finally:
                import os
                os.unlink(tmp_mp3_path)
                if os.path.exists(tmp_wav_path):
                    os.unlink(tmp_wav_path)
        else:
            # Try WAV
            try:
                buffer = io.BytesIO(audio_bytes)
                with wave.open(buffer, 'rb') as wf:
                    audio_sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                    if wf.getnchannels() == 2:
                        audio_data = audio_data.reshape((-1, 2)).mean(axis=1)
            except:
                # Assume raw PCM
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                audio_sr = sample_rate

        # Resample to 16000Hz if needed (Reachy audio device only supports 16000)
        target_sr = 16000
        if audio_sr != target_sr:
            from scipy import signal
            num_samples = int(len(audio_data) * target_sr / audio_sr)
            audio_data = signal.resample(audio_data, num_samples).astype(np.float32)
            audio_sr = target_sr
            print(f"[AUDIO] Resampled to {target_sr}Hz")

        # Convert mono to stereo (Reachy audio device requires 2 channels)
        if len(audio_data.shape) == 1:
            audio_stereo = np.column_stack((audio_data, audio_data))
        else:
            audio_stereo = audio_data

        print(f"[AUDIO] Playing {len(audio_stereo)} samples at {audio_sr}Hz (stereo)")

        # Play on output device - sounddevice infers channels from array shape
        sd.play(audio_stereo, samplerate=audio_sr, device=audio_output_device)
        sd.wait()

        return {"status": "ok", "samples": len(audio_stereo), "duration": len(audio_stereo) / audio_sr}

    except Exception as e:
        print(f"[AUDIO] Play error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/emotion")
async def set_emotion(emotion: str = Form(...), duration: float = Form(1.0)):
    """
    Trigger an emotion expression on Reachy via daemon API.
    Emotions: neutral, happy, sad, angry, thinking, sarcastic, speaking
    """
    global current_emotion
    current_emotion = emotion
    import math
    import httpx

    try:
        print(f"[EMOTION] Expressing: {emotion}")

        # Map emotions to head movements (degrees)
        movements = {
            "neutral": (0, 0, 0),      # pitch, yaw, roll
            "happy": (5, 0, 5),        # slight tilt
            "sad": (-10, 0, 0),        # head down
            "angry": (0, 0, -5),       # slight tilt
            "thinking": (5, 15, 0),    # look up and to side
            "sarcastic": (10, 10, 5),  # head tilt with attitude
            "speaking": (0, 0, 0),     # handled by head wobbler
        }

        pitch_deg, yaw_deg, roll_deg = movements.get(emotion, (0, 0, 0))

        # Convert to radians for daemon API
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)

        # Use the daemon API for head movement
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "head_pose": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                        "roll": roll_rad,
                        "pitch": pitch_rad,
                        "yaw": yaw_rad
                    },
                    "duration": 0.3  # Quick emotion transitions
                },
                timeout=5.0
            )
            result = response.json()
            print(f"[EMOTION] Daemon API response: {result}")

        return {"status": "ok", "emotion": emotion}

    except Exception as e:
        print(f"[EMOTION] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/head/move")
async def move_head(
    pitch: float = Form(0),
    yaw: float = Form(0),
    roll: float = Form(0),
    duration: float = Form(0.5),
):
    """
    Move Reachy's head to specified position via daemon API.
    Angles in radians (daemon API uses radians).
    """
    import math
    import httpx

    try:
        # Clamp to safe ranges (degrees) then convert to radians
        pitch_deg = max(-30, min(30, pitch))
        yaw_deg = max(-45, min(45, yaw))
        roll_deg = max(-15, min(15, roll))

        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)

        print(f"[HEAD] Moving to: pitch={pitch_deg}°, yaw={yaw_deg}°, roll={roll_deg}° (via daemon API)")

        # Use the daemon API for head movement
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "head_pose": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                        "roll": roll_rad,
                        "pitch": pitch_rad,
                        "yaw": yaw_rad
                    },
                    "duration": duration
                },
                timeout=5.0
            )
            result = response.json()
            print(f"[HEAD] Daemon API response: {result}")

        return {"status": "ok", "pitch": pitch_deg, "yaw": yaw_deg, "roll": roll_deg}

    except Exception as e:
        print(f"[HEAD] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/head/wobble")
async def head_wobble(
    intensity: float = Form(0.5),
    duration: float = Form(0.5),
):
    """
    Trigger a speaking wobble animation via daemon API.
    """
    import random
    import math
    import httpx

    try:
        # Quick wobble animation using daemon API
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < duration:
                # Random small movements (in degrees)
                pitch_deg = random.uniform(-3, 3) * intensity
                yaw_deg = random.uniform(-2, 2) * intensity
                roll_deg = random.uniform(-2, 2) * intensity

                # Convert to radians
                pitch_rad = math.radians(pitch_deg)
                yaw_rad = math.radians(yaw_deg)
                roll_rad = math.radians(roll_deg)

                await client.post(
                    f"{DAEMON_API_URL}/api/move/set_target",
                    json={
                        "target_head_pose": {
                            "x": 0,
                            "y": 0,
                            "z": 0,
                            "roll": roll_rad,
                            "pitch": pitch_rad,
                            "yaw": yaw_rad
                        }
                    },
                    timeout=1.0
                )

                await asyncio.sleep(0.1)

            # Return to neutral
            await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "head_pose": {
                        "x": 0, "y": 0, "z": 0,
                        "roll": 0, "pitch": 0, "yaw": 0
                    },
                    "duration": 0.2
                },
                timeout=2.0
            )

        return {"status": "ok"}

    except Exception as e:
        print(f"[WOBBLE] Error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/body/full_move")
async def full_body_move(request: Request):
    """
    Move head, antennas, and body yaw together via daemon API.

    JSON body:
    {
        "head_pose": {"pitch": deg, "yaw": deg, "roll": deg},
        "antennas": [left_rad, right_rad],
        "body_yaw": rad,
        "duration": seconds
    }
    """
    import math
    import httpx

    try:
        data = await request.json()

        head = data.get("head_pose", {})
        pitch_deg = head.get("pitch", 0)
        yaw_deg = head.get("yaw", 0)
        roll_deg = head.get("roll", 0)

        # Clamp head to safe ranges
        pitch_deg = max(-30, min(45, pitch_deg))
        yaw_deg = max(-55, min(55, yaw_deg))
        roll_deg = max(-30, min(30, roll_deg))

        # Convert to radians
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)

        antennas = data.get("antennas", [0.3, 0.3])
        body_yaw = data.get("body_yaw", 0)
        duration = data.get("duration", 0.4)

        # Clamp antennas (-0.5 to 0.9 radians)
        antennas = [max(-0.5, min(0.9, a)) for a in antennas]
        # Clamp body_yaw (-0.5 to 0.5 radians)
        body_yaw = max(-0.5, min(0.5, body_yaw))

        print(f"[BODY] Full move: head=({pitch_deg:.0f}°, {yaw_deg:.0f}°, {roll_deg:.0f}°), "
              f"antennas=[{antennas[0]:.2f}, {antennas[1]:.2f}], body_yaw={body_yaw:.2f}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "head_pose": {
                        "x": 0, "y": 0, "z": 0,
                        "roll": roll_rad,
                        "pitch": pitch_rad,
                        "yaw": yaw_rad
                    },
                    "antennas": antennas,
                    "body_yaw": body_yaw,
                    "duration": duration,
                    "interpolation": "minjerk"
                },
                timeout=5.0
            )
            result = response.json()
            print(f"[BODY] Daemon response: {result}")

        return {
            "status": "ok",
            "head": {"pitch": pitch_deg, "yaw": yaw_deg, "roll": roll_deg},
            "antennas": antennas,
            "body_yaw": body_yaw
        }

    except Exception as e:
        print(f"[BODY] Full move error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/antennas/set")
async def set_antennas(request: Request):
    """
    Set antenna positions.

    JSON body:
    {
        "left": radians,
        "right": radians,
        "duration": seconds
    }
    """
    import httpx

    try:
        data = await request.json()
        left = max(-0.5, min(0.9, data.get("left", 0.3)))
        right = max(-0.5, min(0.9, data.get("right", 0.3)))
        duration = data.get("duration", 0.3)

        print(f"[ANTENNAS] Setting: left={left:.2f}, right={right:.2f}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "antennas": [left, right],
                    "duration": duration
                },
                timeout=5.0
            )
            result = response.json()

        return {"status": "ok", "left": left, "right": right}

    except Exception as e:
        print(f"[ANTENNAS] Error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/antennas/wiggle")
async def wiggle_antennas(
    duration: float = Form(1.0),
    intensity: float = Form(0.5),
):
    """
    Wiggle antennas for specified duration (speaking animation).
    """
    import httpx

    # Wiggle pattern frames
    frames = [
        (0.5, 0.4),
        (0.4, 0.5),
        (0.6, 0.3),
        (0.3, 0.6),
        (0.5, 0.5),
        (0.55, 0.45),
        (0.45, 0.55),
        (0.5, 0.4),
    ]

    try:
        frame_time = duration / len(frames)
        start = time.time()
        frame_idx = 0

        async with httpx.AsyncClient() as client:
            while time.time() - start < duration:
                left, right = frames[frame_idx % len(frames)]
                # Scale by intensity
                left = 0.3 + (left - 0.3) * intensity
                right = 0.3 + (right - 0.3) * intensity

                await client.post(
                    f"{DAEMON_API_URL}/api/move/goto",
                    json={
                        "antennas": [left, right],
                        "duration": frame_time * 0.8
                    },
                    timeout=2.0
                )
                await asyncio.sleep(frame_time)
                frame_idx += 1

            # Return to neutral
            await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "antennas": [0.3, 0.3],
                    "duration": 0.2
                },
                timeout=2.0
            )

        return {"status": "ok", "frames": frame_idx}

    except Exception as e:
        print(f"[ANTENNAS] Wiggle error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/body/yaw")
async def set_body_yaw(
    yaw: float = Form(0),
    duration: float = Form(0.5),
):
    """
    Set body rotation (yaw) in radians.
    """
    import httpx
    import math

    try:
        yaw = max(-0.5, min(0.5, yaw))
        print(f"[BODY] Setting yaw: {yaw:.2f} rad ({math.degrees(yaw):.0f}°)")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_API_URL}/api/move/goto",
                json={
                    "body_yaw": yaw,
                    "duration": duration
                },
                timeout=5.0
            )
            result = response.json()

        return {"status": "ok", "yaw": yaw}

    except Exception as e:
        print(f"[BODY] Yaw error: {e}")
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Reachy Body Server")
    parser.add_argument("--port", type=int, default=8092, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════╗
║           Reachy Body Server                           ║
╠════════════════════════════════════════════════════════╣
║  Audio Endpoints:                                      ║
║    GET  /status           - Server status              ║
║    GET  /audio/capture    - Capture mic audio          ║
║    POST /audio/play_bytes - Play audio on speaker      ║
║  Head Endpoints:                                       ║
║    POST /emotion          - Trigger emotion            ║
║    POST /head/move        - Move head position         ║
║    POST /head/wobble      - Speaking animation         ║
║  Full Body Endpoints:                                  ║
║    POST /body/full_move   - Head + antennas + body     ║
║    POST /body/yaw         - Set body rotation          ║
║    POST /antennas/set     - Set antenna positions      ║
║    POST /antennas/wiggle  - Antenna wiggle animation   ║
╚════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
