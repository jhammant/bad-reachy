"""
Dashboard Server for Bad Reachy
==================================
Web interface to monitor and configure the grumpy robot.
Includes voice cloning and sound effects management.
"""

import threading
import time
import asyncio
import base64
import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
import uvicorn


# Voice and sound effect storage
VOICES_DIR = Path.home() / "bad-reachy-voices"
SOUNDS_DIR = Path.home() / "bad-reachy-sounds"


class BadDashboard:
    """Serves a web dashboard for monitoring and configuring Bad Reachy."""

    def __init__(self, port: int = 8080):
        self.port = port
        self._server_thread = None
        self._app = FastAPI(title="Bad Reachy Dashboard")

        # Create storage directories
        VOICES_DIR.mkdir(exist_ok=True)
        SOUNDS_DIR.mkdir(exist_ok=True)

        # State tracking
        self.state = "IDLE"
        self.last_user_input = ""
        self.last_response = ""
        self.current_emotion = "idle"
        self.conversation_history: List[Dict[str, str]] = []
        self.session_start = time.time()
        self.interactions = 0
        self.swear_count = 0
        self.get_frame = None

        # Voice settings
        self.current_voice = "default"
        self.voices: Dict[str, str] = {"default": None}  # name -> file path
        self.sound_effects: Dict[str, str] = {}  # name -> file path

        # Callbacks for TTS
        self.on_voice_change = None  # Callback when voice changes
        self.on_play_sound = None    # Callback to play sound effect
        self.on_say_text = None      # Callback to say arbitrary text

        # TTS Manager (set by main.py)
        self.tts_manager = None

        self._load_voices()
        self._load_sounds()
        self._setup_routes()

    def _load_voices(self):
        """Load saved voice samples."""
        for f in VOICES_DIR.glob("*.wav"):
            self.voices[f.stem] = str(f)
        for f in VOICES_DIR.glob("*.mp3"):
            self.voices[f.stem] = str(f)

    def _load_sounds(self):
        """Load saved sound effects."""
        for f in SOUNDS_DIR.glob("*.wav"):
            self.sound_effects[f.stem] = str(f)
        for f in SOUNDS_DIR.glob("*.mp3"):
            self.sound_effects[f.stem] = str(f)

        # Add some built-in sound effect descriptions (TTS will generate)
        builtin_sounds = {
            "sigh": "A long, heavy, theatrical sigh",
            "groan": "An annoyed groan",
            "rimshot": "Ba dum tss",
            "crickets": "Awkward cricket sounds",
            "sad_trombone": "Wah wah waaaah",
            "fart": "A comedic fart sound",
            "record_scratch": "Record scratch sound",
        }
        for name, desc in builtin_sounds.items():
            if name not in self.sound_effects:
                self.sound_effects[name] = f"tts:{desc}"

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self._app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()

        @self._app.get("/api/status")
        async def status():
            return {
                "state": self.state,
                "last_user_input": self.last_user_input,
                "last_response": self.last_response,
                "current_emotion": self.current_emotion,
                "session_duration": int(time.time() - self.session_start),
                "interactions": self.interactions,
                "swear_count": self.swear_count,
                "conversation_history": self.conversation_history[-10:],
                "current_voice": self.current_voice,
                "available_voices": list(self.voices.keys()),
                "sound_effects": list(self.sound_effects.keys()),
            }

        @self._app.get("/api/frame")
        async def get_frame():
            if self.get_frame:
                frame = self.get_frame()
                if frame is not None:
                    import cv2
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
            return Response(content=b"", media_type="image/jpeg")

        @self._app.get("/api/frame/stream")
        async def stream_frame():
            def generate():
                while True:
                    if self.get_frame:
                        frame = self.get_frame()
                        if frame is not None:
                            import cv2
                            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.1)
            return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

        # Voice management endpoints
        @self._app.get("/api/voices")
        async def list_voices():
            return {"voices": list(self.voices.keys()), "current": self.current_voice}

        @self._app.post("/api/voices/upload")
        async def upload_voice(name: str = Form(...), file: UploadFile = File(...)):
            """Upload a voice sample for cloning."""
            ext = Path(file.filename).suffix or ".wav"
            filepath = VOICES_DIR / f"{name}{ext}"

            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)

            self.voices[name] = str(filepath)
            return {"status": "ok", "voice": name, "path": str(filepath)}

        @self._app.post("/api/voices/select")
        async def select_voice(name: str = Form(...)):
            """Select a voice to use."""
            if name in self.voices:
                self.current_voice = name
                if self.on_voice_change:
                    self.on_voice_change(name, self.voices.get(name))
                return {"status": "ok", "voice": name}
            return JSONResponse({"error": "Voice not found"}, status_code=404)

        @self._app.delete("/api/voices/{name}")
        async def delete_voice(name: str):
            """Delete a voice sample."""
            if name == "default":
                return JSONResponse({"error": "Cannot delete default voice"}, status_code=400)
            if name in self.voices:
                path = self.voices[name]
                if path and os.path.exists(path):
                    os.remove(path)
                del self.voices[name]
                if self.current_voice == name:
                    self.current_voice = "default"
                return {"status": "ok"}
            return JSONResponse({"error": "Voice not found"}, status_code=404)

        # Sound effects endpoints
        @self._app.get("/api/sounds")
        async def list_sounds():
            return {"sounds": list(self.sound_effects.keys())}

        @self._app.post("/api/sounds/upload")
        async def upload_sound(name: str = Form(...), file: UploadFile = File(...)):
            """Upload a sound effect."""
            ext = Path(file.filename).suffix or ".wav"
            filepath = SOUNDS_DIR / f"{name}{ext}"

            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)

            self.sound_effects[name] = str(filepath)
            return {"status": "ok", "sound": name}

        @self._app.post("/api/sounds/play")
        async def play_sound(name: str = Form(...)):
            """Play a sound effect."""
            if name in self.sound_effects and self.on_play_sound:
                await self.on_play_sound(name, self.sound_effects[name])
                return {"status": "ok", "sound": name}
            return JSONResponse({"error": "Sound not found or player not ready"}, status_code=404)

        @self._app.delete("/api/sounds/{name}")
        async def delete_sound(name: str):
            """Delete a sound effect."""
            if name in self.sound_effects:
                path = self.sound_effects[name]
                if path and not path.startswith("tts:") and os.path.exists(path):
                    os.remove(path)
                del self.sound_effects[name]
                return {"status": "ok"}
            return JSONResponse({"error": "Sound not found"}, status_code=404)

        # Test TTS endpoint
        @self._app.post("/api/test-voice")
        async def test_voice(text: str = Form(default="Testing, testing, one two three. *sigh* This is Bad Reachy.")):
            """Test the current voice."""
            if self.on_play_sound:
                await self.on_play_sound("_test_", f"tts:{text}")
                return {"status": "ok", "text": text}
            return JSONResponse({"error": "TTS not ready"}, status_code=500)

        # Manual say endpoint
        @self._app.post("/api/say")
        async def say_text(text: str = Form(...)):
            """Say arbitrary text through TTS."""
            if self.on_say_text:
                await self.on_say_text(text)
                return {"status": "ok", "text": text}
            return JSONResponse({"error": "TTS not ready"}, status_code=500)

        # TTS Manager endpoints
        @self._app.get("/api/tts/status")
        async def tts_status():
            """Get TTS backend status and available backends."""
            if not self.tts_manager:
                return JSONResponse({"error": "TTS manager not initialized"}, status_code=500)
            return self.tts_manager.get_status()

        @self._app.post("/api/tts/switch")
        async def tts_switch(backend_id: str = Form(...)):
            """Switch to a different TTS backend."""
            if not self.tts_manager:
                return JSONResponse({"error": "TTS manager not initialized"}, status_code=500)
            success = await self.tts_manager.switch_backend(backend_id)
            if success:
                return {"status": "ok", "backend": backend_id}
            return JSONResponse({"error": f"Failed to switch to {backend_id}"}, status_code=400)

        @self._app.post("/api/tts/test")
        async def tts_test(backend_id: str = Form(...)):
            """Test a TTS backend's latency."""
            if not self.tts_manager:
                return JSONResponse({"error": "TTS manager not initialized"}, status_code=500)
            latency = await self.tts_manager.test_backend(backend_id)
            if latency is not None:
                return {"status": "ok", "backend_id": backend_id, "latency_ms": latency}
            return JSONResponse({"error": f"Test failed for {backend_id}"}, status_code=400)

        @self._app.post("/api/tts/benchmark")
        async def tts_benchmark():
            """Benchmark all available TTS backends."""
            if not self.tts_manager:
                return JSONResponse({"error": "TTS manager not initialized"}, status_code=500)
            results = await self.tts_manager.benchmark_all()
            return {"status": "ok", "results": results}

        @self._app.post("/api/tts/remote-server")
        async def tts_set_remote(url: str = Form(...)):
            """Set the remote TTS server URL."""
            if not self.tts_manager:
                return JSONResponse({"error": "TTS manager not initialized"}, status_code=500)
            self.tts_manager.set_remote_server(url)
            return {"status": "ok", "url": url}

    def update_state(self, state: str):
        self.state = state

    def update_emotion(self, emotion: str):
        self.current_emotion = emotion

    def add_interaction(self, user_input: str, response: str):
        self.last_user_input = user_input
        self.last_response = response
        self.interactions += 1
        self.conversation_history.append({
            "user": user_input,
            "bad": response,
            "time": time.strftime("%H:%M:%S")
        })

        swears = ["fuck", "shit", "damn", "hell", "ass", "crap", "bastard", "bitch"]
        for swear in swears:
            self.swear_count += response.lower().count(swear)

    def _get_dashboard_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bad Reachy - Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a0a0a 0%, #2d1a1a 50%, #1a1a2e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,0,0,0.05);
            border-radius: 16px;
            border: 1px solid rgba(255,100,100,0.2);
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #ff4444, #ff8800);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #888; font-size: 1.1em; }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 1em;
        }
        .tab.active {
            background: linear-gradient(135deg, #ff4444, #ff8800);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            color: #ff6b6b;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .camera-feed {
            width: 100%;
            border-radius: 12px;
            background: #000;
            aspect-ratio: 4/3;
        }
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .status-item {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        .status-label { color: #888; font-size: 0.85em; margin-bottom: 5px; }
        .status-value { font-size: 1.4em; font-weight: 600; }
        .status-value.active { color: #ff4444; }
        .emotion-badge {
            display: inline-block;
            padding: 8px 16px;
            background: linear-gradient(135deg, #ff4444, #ff8800);
            border-radius: 20px;
            font-size: 1.2em;
            text-transform: uppercase;
        }
        .conversation {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
        }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
        .message.user { background: rgba(100,100,255,0.2); border-left: 3px solid #6666ff; }
        .message.bad { background: rgba(255,100,100,0.2); border-left: 3px solid #ff4444; }
        .last-response {
            background: rgba(255,100,100,0.1);
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #ff4444;
            font-style: italic;
            min-height: 60px;
        }
        /* Voice & Sounds tab */
        .voice-list, .sound-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }
        .voice-item, .sound-item {
            padding: 10px 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .voice-item.active { background: linear-gradient(135deg, #ff4444, #ff8800); }
        .voice-item:hover, .sound-item:hover { background: rgba(255,255,255,0.2); }
        .upload-form {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 12px;
            margin-top: 15px;
        }
        .upload-form input[type="text"], .upload-form input[type="file"] {
            width: 100%;
            padding: 10px;
            margin: 5px 0 15px 0;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(0,0,0,0.3);
            color: #fff;
        }
        .btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #ff4444, #ff8800);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover { opacity: 0.9; }
        .btn-secondary { background: rgba(255,255,255,0.2); }
        .btn-small { padding: 5px 10px; font-size: 0.9em; }
        .swear-counter { font-size: 2.5em; color: #ff4444; }
        /* TTS Backend styles */
        .backend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px;
            margin-bottom: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s;
        }
        .backend-item:hover { background: rgba(255,255,255,0.1); }
        .backend-item.active { border-color: #ff6b6b; background: rgba(255,100,100,0.15); }
        .backend-item.unavailable { opacity: 0.5; cursor: not-allowed; }
        .backend-item.loading { opacity: 0.7; cursor: wait; }
        .backend-info { flex: 1; }
        .backend-name { font-weight: 600; font-size: 1.1em; display: flex; align-items: center; gap: 8px; }
        .backend-desc { color: #888; font-size: 0.9em; margin-top: 4px; }
        .backend-actions { display: flex; align-items: center; gap: 10px; }
        .backend-latency { color: #4ecdc4; font-size: 0.9em; min-width: 60px; text-align: right; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }
        .badge.local { background: #2d5a27; color: #8fd88d; }
        .badge.cloud { background: #1a3a5c; color: #7db8e8; }
        .badge.loaded { background: #5a2d27; color: #ff8888; }
        .loading-overlay {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: rgba(255,100,100,0.1);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #ff6b6b;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .platform-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .platform-item:last-child { border-bottom: none; }
        .platform-label { color: #888; }
        .platform-value { font-weight: 600; }
        .platform-value.yes { color: #4ecdc4; }
        .platform-value.no { color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Bad Reachy</h1>
            <p class="subtitle">Sarcastic Robot Comedian with Attitude</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="showTab('monitor')">Monitor</button>
            <button class="tab" onclick="showTab('voices')">Voices & Sounds</button>
            <button class="tab" onclick="showTab('settings')">Settings</button>
        </div>

        <!-- Monitor Tab -->
        <div id="tab-monitor" class="tab-content active">
            <div class="grid">
                <div class="card">
                    <h2>Camera View</h2>
                    <img id="camera-feed" class="camera-feed" src="/api/frame/stream" alt="Camera">
                </div>
                <div class="card">
                    <h2>Status</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="status-label">State</div>
                            <div class="status-value" id="app-state">IDLE</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Session</div>
                            <div class="status-value" id="session-time">00:00</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Interactions</div>
                            <div class="status-value" id="interactions">0</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Swear Count</div>
                            <div class="status-value swear-counter" id="swear-count">0</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; text-align: center;">
                        <div class="status-label">Emotion</div>
                        <div class="emotion-badge" id="emotion">idle</div>
                    </div>
                </div>
                <div class="card">
                    <h2>🎤 What I Heard</h2>
                    <div class="last-input" id="last-input" style="font-size: 1.2em; color: #4ecdc4; min-height: 30px;">Listening...</div>
                </div>
                <div class="card">
                    <h2>💬 Last Response</h2>
                    <div class="last-response" id="last-response">Waiting...</div>
                </div>
                <div class="card">
                    <h2>🗣️ Manual Say</h2>
                    <p style="color: #888; margin-bottom: 10px;">Type text to make Reachy say it</p>
                    <textarea id="say-text" rows="3" style="width: 100%; padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #fff; font-size: 1em; resize: vertical;" placeholder="Type something for Reachy to say... Use [INNER]text[/INNER] for inner voice"></textarea>
                    <div style="margin-top: 10px; display: flex; gap: 10px;">
                        <button class="btn" onclick="sayText()">Say This</button>
                        <button class="btn btn-secondary" onclick="document.getElementById('say-text').value = ''">Clear</button>
                    </div>
                </div>
                <div class="card">
                    <h2>📜 Conversation</h2>
                    <div class="conversation" id="conversation"></div>
                </div>
            </div>
        </div>

        <!-- Voices & Sounds Tab -->
        <div id="tab-voices" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h2>Voice Clones</h2>
                    <p style="color: #888; margin-bottom: 10px;">Upload voice samples to clone different characters</p>
                    <div class="voice-list" id="voice-list"></div>
                    <div class="upload-form">
                        <label>Voice Name:</label>
                        <input type="text" id="voice-name" placeholder="e.g., morgan_freeman">
                        <label>Audio Sample (WAV/MP3):</label>
                        <input type="file" id="voice-file" accept="audio/*">
                        <button class="btn" onclick="uploadVoice()">Upload Voice</button>
                        <button class="btn btn-secondary" onclick="testVoice()">Test Current Voice</button>
                    </div>
                </div>
                <div class="card">
                    <h2>Sound Effects</h2>
                    <p style="color: #888; margin-bottom: 10px;">Sound effects for comedy timing</p>
                    <div class="sound-list" id="sound-list"></div>
                    <div class="upload-form">
                        <label>Sound Name:</label>
                        <input type="text" id="sound-name" placeholder="e.g., explosion">
                        <label>Audio File:</label>
                        <input type="file" id="sound-file" accept="audio/*">
                        <button class="btn" onclick="uploadSound()">Upload Sound</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Settings Tab -->
        <div id="tab-settings" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h2>TTS Backend</h2>
                    <p style="color: #888; margin-bottom: 15px;">Select text-to-speech engine for voice synthesis</p>
                    <div id="tts-loading" class="loading-overlay" style="display: none;">
                        <div class="spinner"></div>
                        <span>Loading model...</span>
                    </div>
                    <div id="tts-backend-list"></div>
                    <div style="margin-top: 20px;">
                        <button class="btn" onclick="benchmarkTTS()" id="benchmark-btn">Benchmark All</button>
                        <span id="benchmark-status" style="margin-left: 10px; color: #888;"></span>
                    </div>
                </div>
                <div class="card">
                    <h2>Platform Info</h2>
                    <div id="platform-info">
                        <div class="platform-item">
                            <span class="platform-label">System:</span>
                            <span class="platform-value" id="platform-system">-</span>
                        </div>
                        <div class="platform-item">
                            <span class="platform-label">Architecture:</span>
                            <span class="platform-value" id="platform-arch">-</span>
                        </div>
                        <div class="platform-item">
                            <span class="platform-label">Apple Silicon:</span>
                            <span class="platform-value" id="platform-apple">-</span>
                        </div>
                        <div class="platform-item">
                            <span class="platform-label">CUDA:</span>
                            <span class="platform-value" id="platform-cuda">-</span>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h2>Remote TTS Server</h2>
                    <p style="color: #888; margin-bottom: 15px;">Run TTS inference on your Mac and stream audio to Reachy</p>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="text" id="remote-server-url" placeholder="http://192.168.1.100:8090"
                               style="flex: 1; padding: 10px; border: 1px solid #333; border-radius: 4px; background: #1a1a1a; color: #fff;">
                        <button class="btn" onclick="setRemoteServer()">Set</button>
                        <button class="btn" onclick="testRemoteServer()" style="background: #333;">Test</button>
                    </div>
                    <div id="remote-server-status" style="margin-top: 10px; color: #888;"></div>
                    <p style="color: #666; font-size: 12px; margin-top: 10px;">
                        Run <code style="background: #333; padding: 2px 6px; border-radius: 3px;">python tts_server.py --host 0.0.0.0</code> on your Mac to start the server.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tab}')"]`).classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
        }

        function formatTime(s) {
            return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toString().padStart(2,'0')}`;
        }

        async function updateStatus() {
            try {
                const r = await fetch('/api/status');
                const d = await r.json();
                document.getElementById('app-state').textContent = d.state;
                document.getElementById('session-time').textContent = formatTime(d.session_duration);
                document.getElementById('interactions').textContent = d.interactions;
                document.getElementById('swear-count').textContent = d.swear_count;
                document.getElementById('emotion').textContent = d.current_emotion;
                document.getElementById('last-input').textContent = d.last_user_input || (d.state === 'LISTENING' ? '🎧 Listening...' : 'Waiting for speech...');
                document.getElementById('last-response').textContent = d.last_response || 'Waiting...';

                const conv = document.getElementById('conversation');
                conv.innerHTML = d.conversation_history.map(c => `
                    <div class="message user"><strong>You:</strong> ${c.user}</div>
                    <div class="message bad"><strong>Bad:</strong> ${c.bad}</div>
                `).join('');
                conv.scrollTop = conv.scrollHeight;

                // Update voice list
                const vl = document.getElementById('voice-list');
                vl.innerHTML = d.available_voices.map(v => `
                    <div class="voice-item ${v === d.current_voice ? 'active' : ''}" onclick="selectVoice('${v}')">
                        ${v} ${v === d.current_voice ? '✓' : ''}
                    </div>
                `).join('');

                // Update sound list
                const sl = document.getElementById('sound-list');
                sl.innerHTML = d.sound_effects.map(s => `
                    <div class="sound-item" onclick="playSound('${s}')">${s}</div>
                `).join('');
            } catch(e) { console.error(e); }
        }

        async function selectVoice(name) {
            const fd = new FormData();
            fd.append('name', name);
            await fetch('/api/voices/select', {method: 'POST', body: fd});
            updateStatus();
        }

        async function uploadVoice() {
            const name = document.getElementById('voice-name').value;
            const file = document.getElementById('voice-file').files[0];
            if (!name || !file) return alert('Name and file required');
            const fd = new FormData();
            fd.append('name', name);
            fd.append('file', file);
            await fetch('/api/voices/upload', {method: 'POST', body: fd});
            document.getElementById('voice-name').value = '';
            updateStatus();
        }

        async function testVoice() {
            const fd = new FormData();
            fd.append('text', 'Oh great, another test. *sigh* This is Bad Reachy speaking.');
            await fetch('/api/test-voice', {method: 'POST', body: fd});
        }

        async function sayText() {
            const text = document.getElementById('say-text').value.trim();
            if (!text) return alert('Enter some text first');
            const fd = new FormData();
            fd.append('text', text);
            const r = await fetch('/api/say', {method: 'POST', body: fd});
            if (r.ok) {
                document.getElementById('say-text').style.borderColor = '#4ecdc4';
                setTimeout(() => document.getElementById('say-text').style.borderColor = 'rgba(255,255,255,0.2)', 1000);
            }
        }

        async function playSound(name) {
            const fd = new FormData();
            fd.append('name', name);
            await fetch('/api/sounds/play', {method: 'POST', body: fd});
        }

        async function uploadSound() {
            const name = document.getElementById('sound-name').value;
            const file = document.getElementById('sound-file').files[0];
            if (!name || !file) return alert('Name and file required');
            const fd = new FormData();
            fd.append('name', name);
            fd.append('file', file);
            await fetch('/api/sounds/upload', {method: 'POST', body: fd});
            document.getElementById('sound-name').value = '';
            updateStatus();
        }

        // TTS Manager functions
        let ttsStatus = null;

        async function updateTTSStatus() {
            try {
                const r = await fetch('/api/tts/status');
                if (!r.ok) return;
                ttsStatus = await r.json();
                renderTTSBackends();
                renderPlatformInfo();
            } catch(e) { console.error('TTS status error:', e); }
        }

        function renderTTSBackends() {
            if (!ttsStatus) return;
            const list = document.getElementById('tts-backend-list');
            if (!list) return;

            const loading = document.getElementById('tts-loading');
            if (loading) {
                loading.style.display = ttsStatus.is_loading ? 'flex' : 'none';
            }

            list.innerHTML = ttsStatus.backends.map(b => {
                const isActive = b.id === ttsStatus.current_backend;
                const classes = ['backend-item'];
                if (isActive) classes.push('active');
                if (!b.available) classes.push('unavailable');
                if (ttsStatus.is_loading && isActive) classes.push('loading');

                const badges = [];
                badges.push(`<span class="badge ${b.is_local ? 'local' : 'cloud'}">${b.is_local ? 'Local' : 'Cloud'}</span>`);
                if (b.loaded) badges.push('<span class="badge loaded">Loaded</span>');

                const latency = b.last_latency_ms ? `${Math.round(b.last_latency_ms)}ms` : '';

                return `
                    <div class="${classes.join(' ')}" onclick="${b.available ? `switchTTSBackend('${b.id}')` : ''}">
                        <div class="backend-info">
                            <div class="backend-name">
                                ${isActive ? '●' : '○'} ${b.name}
                                ${badges.join(' ')}
                            </div>
                            <div class="backend-desc">${b.backend_type} - ${b.description}</div>
                        </div>
                        <div class="backend-actions">
                            <span class="backend-latency">${latency}</span>
                            ${b.available && !isActive ? `<button class="btn btn-small btn-secondary" onclick="event.stopPropagation(); testTTSBackend('${b.id}')">Test</button>` : ''}
                            ${isActive ? '✓' : ''}
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderPlatformInfo() {
            if (!ttsStatus || !ttsStatus.platform) return;
            const p = ttsStatus.platform;
            document.getElementById('platform-system').textContent = p.system;
            document.getElementById('platform-arch').textContent = p.machine;
            const appleEl = document.getElementById('platform-apple');
            appleEl.textContent = p.is_apple_silicon ? 'Yes' : 'No';
            appleEl.className = 'platform-value ' + (p.is_apple_silicon ? 'yes' : 'no');
            const cudaEl = document.getElementById('platform-cuda');
            cudaEl.textContent = p.has_cuda ? 'Yes' : 'No';
            cudaEl.className = 'platform-value ' + (p.has_cuda ? 'yes' : 'no');
        }

        async function switchTTSBackend(backendId) {
            if (ttsStatus && ttsStatus.is_loading) return;
            const loading = document.getElementById('tts-loading');
            if (loading) loading.style.display = 'flex';

            const fd = new FormData();
            fd.append('backend_id', backendId);
            const r = await fetch('/api/tts/switch', {method: 'POST', body: fd});
            if (r.ok) {
                await updateTTSStatus();
            } else {
                alert('Failed to switch backend');
                if (loading) loading.style.display = 'none';
            }
        }

        async function testTTSBackend(backendId) {
            const fd = new FormData();
            fd.append('backend_id', backendId);
            const r = await fetch('/api/tts/test', {method: 'POST', body: fd});
            if (r.ok) {
                await updateTTSStatus();
            } else {
                alert('Test failed');
            }
        }

        async function benchmarkTTS() {
            const btn = document.getElementById('benchmark-btn');
            const status = document.getElementById('benchmark-status');
            btn.disabled = true;
            status.textContent = 'Running benchmarks...';

            const r = await fetch('/api/tts/benchmark', {method: 'POST'});
            if (r.ok) {
                const data = await r.json();
                status.textContent = `Completed! Fastest: ${data.results[0]?.name || 'N/A'}`;
                await updateTTSStatus();
            } else {
                status.textContent = 'Benchmark failed';
            }
            btn.disabled = false;
        }

        // Remote TTS Server functions
        function updateRemoteServerUI() {
            if (ttsStatus && ttsStatus.remote_server_url) {
                document.getElementById('remote-server-url').value = ttsStatus.remote_server_url;
            }
        }

        async function setRemoteServer() {
            const url = document.getElementById('remote-server-url').value.trim();
            if (!url) return alert('Please enter a URL');

            const status = document.getElementById('remote-server-status');
            status.textContent = 'Setting remote server...';
            status.style.color = '#888';

            const fd = new FormData();
            fd.append('url', url);
            const r = await fetch('/api/tts/remote-server', {method: 'POST', body: fd});
            if (r.ok) {
                status.textContent = 'Remote server URL updated! Select "Remote Mac TTS" in backends above.';
                status.style.color = '#4ecdc4';
                await updateTTSStatus();
            } else {
                status.textContent = 'Failed to set remote server';
                status.style.color = '#ff6b6b';
            }
        }

        async function testRemoteServer() {
            const url = document.getElementById('remote-server-url').value.trim();
            if (!url) return alert('Please enter a URL');

            const status = document.getElementById('remote-server-status');
            status.textContent = 'Testing connection...';
            status.style.color = '#888';

            try {
                const r = await fetch(url + '/status', {method: 'GET', mode: 'cors'});
                if (r.ok) {
                    const data = await r.json();
                    status.innerHTML = `Connected! Server status: ${data.status}<br>Model: ${data.current_model || 'N/A'}<br>Latency: ${data.last_latency_ms ? Math.round(data.last_latency_ms) + 'ms' : 'N/A'}`;
                    status.style.color = '#4ecdc4';
                } else {
                    status.textContent = 'Server responded with error: ' + r.status;
                    status.style.color = '#ff6b6b';
                }
            } catch(e) {
                status.textContent = 'Connection failed: ' + e.message;
                status.style.color = '#ff6b6b';
            }
        }

        setInterval(updateStatus, 1000);
        setInterval(updateTTSStatus, 2000);
        updateStatus();
        updateTTSStatus().then(updateRemoteServerUI);
        document.getElementById('camera-feed').onerror = function() {
            setTimeout(() => { this.src = '/api/frame/stream?' + Date.now(); }, 1000);
        };
    </script>
</body>
</html>'''

    def start(self):
        """Start the dashboard server."""
        def run_server():
            try:
                print(f"[DASHBOARD] Starting uvicorn on port {self.port}...")
                uvicorn.run(self._app, host="0.0.0.0", port=self.port, log_level="info")
            except Exception as e:
                print(f"[DASHBOARD] Server crashed: {e}")
                import traceback
                traceback.print_exc()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to bind
        time.sleep(1.0)

        print(f"[DASHBOARD] Started on http://0.0.0.0:{self.port}")
