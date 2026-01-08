"""
Dashboard Server for Bad Reachy
==================================
Web interface to monitor the grumpy robot.
"""

import threading
import time
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import uvicorn


class BadDashboard:
    """Serves a web dashboard for monitoring Bad Reachy."""

    def __init__(self, port: int = 8080):
        self.port = port
        self._server_thread = None
        self._app = FastAPI(title="Bad Reachy Dashboard")

        # State tracking
        self.state = "IDLE"
        self.last_user_input = ""
        self.last_response = ""
        self.current_emotion = "idle"
        self.conversation_history: List[Dict[str, str]] = []
        self.session_start = time.time()
        self.interactions = 0
        self.swear_count = 0
        self.get_frame = None  # Camera frame getter

        self._setup_routes()

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
            "grumpy": response,
            "time": time.strftime("%H:%M:%S")
        })

        # Count swears for fun stats
        swears = ["fuck", "shit", "damn", "hell", "ass", "crap", "bastard"]
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
        .container { max-width: 1200px; margin: 0 auto; }
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
        .status-value.idle { color: #666; }
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
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .message.user {
            background: rgba(100,100,255,0.2);
            border-left: 3px solid #6666ff;
        }
        .message.grumpy {
            background: rgba(255,100,100,0.2);
            border-left: 3px solid #ff4444;
        }
        .message-time { font-size: 0.75em; color: #666; }
        .swear-counter {
            font-size: 2em;
            color: #ff4444;
            text-align: center;
            padding: 20px;
        }
        .last-response {
            background: rgba(255,100,100,0.1);
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #ff4444;
            font-style: italic;
            min-height: 60px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Bad Reachy</h1>
            <p class="subtitle">Your favorite cynical robot assistant</p>
        </header>

        <div class="grid">
            <div class="card">
                <h2>Camera View</h2>
                <img id="camera-feed" class="camera-feed" src="/api/frame/stream" alt="Camera Feed">
            </div>

            <div class="card">
                <h2>Status</h2>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">State</div>
                        <div class="status-value" id="app-state">IDLE</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Session Time</div>
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
                    <div class="status-label">Current Emotion</div>
                    <div class="emotion-badge" id="emotion">idle</div>
                </div>
            </div>

            <div class="card">
                <h2>Last Response</h2>
                <div class="last-response" id="last-response">Waiting for someone to bother me...</div>
            </div>

            <div class="card">
                <h2>Conversation History</h2>
                <div class="conversation" id="conversation"></div>
            </div>
        </div>
    </div>

    <script>
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                document.getElementById('app-state').textContent = data.state;
                document.getElementById('app-state').className = 'status-value ' +
                    (data.state === 'LISTENING' ? 'active' : 'idle');
                document.getElementById('session-time').textContent = formatTime(data.session_duration);
                document.getElementById('interactions').textContent = data.interactions;
                document.getElementById('swear-count').textContent = data.swear_count;
                document.getElementById('emotion').textContent = data.current_emotion;
                document.getElementById('last-response').textContent =
                    data.last_response || 'Waiting for someone to bother me...';

                // Update conversation
                const convEl = document.getElementById('conversation');
                convEl.innerHTML = data.conversation_history.map(c => `
                    <div class="message user">
                        <div class="message-time">${c.time}</div>
                        <strong>You:</strong> ${c.user}
                    </div>
                    <div class="message grumpy">
                        <strong>Grumpy:</strong> ${c.grumpy}
                    </div>
                `).join('');
                convEl.scrollTop = convEl.scrollHeight;

            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        setInterval(updateStatus, 1000);
        updateStatus();

        document.getElementById('camera-feed').onerror = function() {
            setTimeout(() => { this.src = '/api/frame/stream?' + Date.now(); }, 1000);
        };
    </script>
</body>
</html>'''

    def start(self):
        """Start the dashboard server in a background thread."""
        def run_server():
            uvicorn.run(self._app, host="0.0.0.0", port=self.port, log_level="warning")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        print(f"[DASHBOARD] Started on http://0.0.0.0:{self.port}")
