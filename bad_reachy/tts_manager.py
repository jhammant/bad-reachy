"""
TTS Manager for Bad Reachy
===========================
Runtime TTS backend switching with lazy loading and benchmarking.
"""

import asyncio
import platform
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


@dataclass
class TTSBackendInfo:
    """Information about a TTS backend."""
    id: str                      # "mlx-kokoro", "edge", etc.
    name: str                    # Display name
    backend_type: str            # "mlx", "qwen3", "edge"
    model: Optional[str]         # Model variant
    available: bool              # Platform-available?
    loaded: bool                 # Currently loaded?
    last_latency_ms: Optional[float] = None
    is_local: bool = True
    description: str = ""


class TTSManager:
    """
    Manages TTS backends with lazy loading and runtime switching.

    Supports:
    - MLX-Audio models (Mac only): kokoro, marvis, qwen3, csm, chatterbox
    - Qwen3-TTS (CUDA): 0.6B and 1.7B models
    - EdgeTTS (cloud): Microsoft's free TTS
    - Remote TTS: Call a TTS server running on another machine (e.g., Mac)
    """

    def __init__(self, remote_server_url: Optional[str] = None):
        self._current_backend_id: Optional[str] = None
        self._current_tts = None
        self._is_loading = False
        self._backends: Dict[str, TTSBackendInfo] = {}
        self._latencies: Dict[str, float] = {}

        # Remote server configuration
        import os
        self._remote_server_url = remote_server_url or os.environ.get(
            "TTS_SERVER_URL", "http://192.168.1.100:8090"  # Default to local network
        )

        # Platform detection
        self._is_mac = platform.system() == "Darwin"
        self._is_apple_silicon = self._is_mac and platform.machine() == "arm64"
        self._has_cuda = self._check_cuda()

        # Detect available backends
        self._detect_backends()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _detect_backends(self):
        """Detect all available TTS backends on this platform."""
        # MLX backends (Apple Silicon only)
        mlx_available = self._is_apple_silicon and self._check_mlx_available()

        if mlx_available:
            mlx_models = [
                ("mlx-spark", "Spark 0.5B", "spark", "Fast local TTS (~5s)"),
                ("mlx-marvis", "Marvis 250M", "marvis", "Streaming capable (~30s)"),
                ("mlx-kokoro", "Kokoro 82M", "kokoro", "Fastest (requires Python <3.14)"),
            ]
            for backend_id, name, model, desc in mlx_models:
                self._backends[backend_id] = TTSBackendInfo(
                    id=backend_id,
                    name=name,
                    backend_type="mlx",
                    model=model,
                    available=True,
                    loaded=False,
                    is_local=True,
                    description=desc,
                )
        else:
            # Show MLX backends as unavailable on non-Mac
            mlx_models = [
                ("mlx-spark", "Spark 0.5B", "spark", "Fast local TTS (Mac only)"),
                ("mlx-marvis", "Marvis 250M", "marvis", "Streaming capable (Mac only)"),
            ]
            for backend_id, name, model, desc in mlx_models:
                self._backends[backend_id] = TTSBackendInfo(
                    id=backend_id,
                    name=name,
                    backend_type="mlx",
                    model=model,
                    available=False,
                    loaded=False,
                    is_local=True,
                    description=desc,
                )

        # Qwen3-TTS backends (CUDA only)
        qwen3_available = self._has_cuda and self._check_qwen3_available()

        qwen3_models = [
            ("qwen3-cuda-0.6b", "Qwen3 0.6B CUDA", "0.6B", "Ultra-low latency (~97ms)"),
            ("qwen3-cuda-1.7b", "Qwen3 1.7B CUDA", "1.7B", "Higher quality"),
        ]
        for backend_id, name, model, desc in qwen3_models:
            self._backends[backend_id] = TTSBackendInfo(
                id=backend_id,
                name=name,
                backend_type="qwen3",
                model=model,
                available=qwen3_available,
                loaded=False,
                is_local=True,
                description=desc if qwen3_available else f"{desc} (CUDA only)",
            )

        # EdgeTTS (always available if package installed)
        edge_available = self._check_edge_available()
        self._backends["edge"] = TTSBackendInfo(
            id="edge",
            name="Edge TTS",
            backend_type="edge",
            model=None,
            available=edge_available,
            loaded=False,
            is_local=False,
            description="Microsoft cloud TTS - reliable fallback",
        )

        # Remote TTS (always shown, availability checked on connect)
        self._backends["remote"] = TTSBackendInfo(
            id="remote",
            name="Remote Mac TTS",
            backend_type="remote",
            model=None,
            available=True,  # Assume available, will fail if server not running
            loaded=False,
            is_local=False,
            description=f"MLX TTS on Mac ({self._remote_server_url})",
        )

    def _check_mlx_available(self) -> bool:
        """Check if MLX-Audio is available."""
        try:
            from mlx_audio.tts.utils import load_model
            return True
        except ImportError:
            return False

    def _check_qwen3_available(self) -> bool:
        """Check if Qwen3-TTS is available."""
        try:
            from qwen_tts import Qwen3TTSModel
            return True
        except ImportError:
            return False

    def _check_edge_available(self) -> bool:
        """Check if EdgeTTS is available."""
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def detect_available_backends(self) -> List[TTSBackendInfo]:
        """Return list of all backends with availability status."""
        return list(self._backends.values())

    def get_status(self) -> Dict[str, Any]:
        """Get current TTS status for API response."""
        backends = []
        for backend in self._backends.values():
            backends.append({
                "id": backend.id,
                "name": backend.name,
                "backend_type": backend.backend_type,
                "model": backend.model,
                "available": backend.available,
                "loaded": backend.loaded,
                "last_latency_ms": self._latencies.get(backend.id),
                "is_local": backend.is_local,
                "description": backend.description,
            })

        return {
            "current_backend": self._current_backend_id,
            "is_loading": self._is_loading,
            "backends": backends,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "is_apple_silicon": self._is_apple_silicon,
                "has_cuda": self._has_cuda,
            },
            "remote_server_url": self._remote_server_url,
        }

    def set_remote_server(self, url: str):
        """Update the remote TTS server URL."""
        self._remote_server_url = url.rstrip('/')
        # Update the backend description
        if "remote" in self._backends:
            self._backends["remote"].description = f"MLX TTS on Mac ({self._remote_server_url})"
        print(f"[TTS-MGR] Remote server set to: {self._remote_server_url}")

    async def switch_backend(self, backend_id: str) -> bool:
        """
        Switch to a different TTS backend.
        Returns True on success, False on failure.
        """
        if backend_id not in self._backends:
            print(f"[TTS-MGR] Unknown backend: {backend_id}")
            return False

        backend_info = self._backends[backend_id]
        if not backend_info.available:
            print(f"[TTS-MGR] Backend not available: {backend_id}")
            return False

        if self._current_backend_id == backend_id and self._current_tts is not None:
            print(f"[TTS-MGR] Already using {backend_id}")
            return True

        self._is_loading = True
        print(f"[TTS-MGR] Switching to {backend_id}...")

        try:
            # Mark old backend as unloaded
            if self._current_backend_id and self._current_backend_id in self._backends:
                self._backends[self._current_backend_id].loaded = False

            # Load new backend
            new_tts = await self._load_backend(backend_id)
            if new_tts is None:
                self._is_loading = False
                return False

            # Unload old TTS to free memory
            self._current_tts = None

            # Set new backend
            self._current_tts = new_tts
            self._current_backend_id = backend_id
            self._backends[backend_id].loaded = True

            print(f"[TTS-MGR] Switched to {backend_id}")
            self._is_loading = False
            return True

        except Exception as e:
            print(f"[TTS-MGR] Failed to switch to {backend_id}: {e}")
            import traceback
            traceback.print_exc()
            self._is_loading = False
            return False

    async def _load_backend(self, backend_id: str):
        """Load a TTS backend by ID and verify it works."""
        backend_info = self._backends[backend_id]
        tts = None

        try:
            if backend_info.backend_type == "mlx":
                from .tts_fast import MLXAudioTTS
                tts = MLXAudioTTS(
                    model=backend_info.model,
                    speed=1.1,
                    language="English",
                )

            elif backend_info.backend_type == "qwen3":
                from .tts_fast import Qwen3TTS
                tts = Qwen3TTS(
                    model_size=backend_info.model,
                    voice_mode="custom",
                    speaker="Ethan",
                    language="English",
                )

            elif backend_info.backend_type == "edge":
                from .tts_fast import EdgeTTS
                tts = EdgeTTS(
                    voice="en-GB-RyanNeural",
                    rate="+15%",
                    pitch="-5Hz",
                )

            elif backend_info.backend_type == "remote":
                from .tts_fast import RemoteTTS
                tts = RemoteTTS(server_url=self._remote_server_url)

            # Verify the backend actually works before returning
            if tts is not None:
                print(f"[TTS-MGR] Testing {backend_id}...")
                if await tts.test_connection():
                    print(f"[TTS-MGR] {backend_id} verified working")
                    return tts
                else:
                    print(f"[TTS-MGR] {backend_id} failed test - not usable")
                    return None

        except Exception as e:
            print(f"[TTS-MGR] Failed to load {backend_id}: {e}")
            return None

        return None

    async def test_backend(self, backend_id: str) -> Optional[float]:
        """
        Test a backend's latency.
        Returns latency in ms, or None on failure.
        """
        if backend_id not in self._backends:
            return None

        backend_info = self._backends[backend_id]
        if not backend_info.available:
            return None

        try:
            # If this is the current backend, use it directly
            if backend_id == self._current_backend_id and self._current_tts:
                tts = self._current_tts
            else:
                # Load temporarily for testing
                tts = await self._load_backend(backend_id)
                if tts is None:
                    return None

            # Measure latency
            test_text = "Testing one two three."
            start = time.time()
            audio = await tts.synthesize(test_text)
            latency_ms = (time.time() - start) * 1000

            if audio is None:
                return None

            # Store latency
            self._latencies[backend_id] = latency_ms

            return latency_ms

        except Exception as e:
            print(f"[TTS-MGR] Test failed for {backend_id}: {e}")
            return None

    async def benchmark_all(self) -> List[Dict[str, Any]]:
        """
        Benchmark all available backends.
        Returns list of results with backend_id, latency_ms, success.
        """
        results = []

        for backend_id, backend_info in self._backends.items():
            if not backend_info.available:
                results.append({
                    "backend_id": backend_id,
                    "name": backend_info.name,
                    "latency_ms": None,
                    "success": False,
                    "error": "Not available on this platform",
                })
                continue

            print(f"[TTS-MGR] Benchmarking {backend_id}...")
            latency = await self.test_backend(backend_id)

            results.append({
                "backend_id": backend_id,
                "name": backend_info.name,
                "latency_ms": latency,
                "success": latency is not None,
                "error": None if latency else "Synthesis failed",
            })

        # Sort by latency (fastest first)
        results.sort(key=lambda x: x["latency_ms"] if x["latency_ms"] else float('inf'))

        return results

    def get_current_backend(self):
        """Get the current TTS instance."""
        return self._current_tts

    def outputs_wav(self) -> bool:
        """Check if current backend outputs WAV (vs MP3 for EdgeTTS)."""
        if self._current_backend_id is None:
            return True
        backend_type = self._backends[self._current_backend_id].backend_type
        # Remote server uses EdgeTTS by default (returns MP3)
        # Edge returns MP3, all others return WAV
        return backend_type not in ("edge", "remote")

    def set_preferred_backend(self, preference: str):
        """
        Set the preferred backend based on environment variable.
        Preference can be: "mlx", "qwen3", "edge", or "auto"
        """
        preference = preference.lower()

        if preference == "auto":
            # Priority: mlx (Mac) -> qwen3 (CUDA) -> edge (cloud)
            if self._is_apple_silicon:
                for backend_id in ["mlx-kokoro", "mlx-marvis"]:
                    if self._backends.get(backend_id, TTSBackendInfo("", "", "", None, False, False)).available:
                        asyncio.create_task(self.switch_backend(backend_id))
                        return

            if self._has_cuda:
                for backend_id in ["qwen3-cuda-0.6b", "qwen3-cuda-1.7b"]:
                    if self._backends.get(backend_id, TTSBackendInfo("", "", "", None, False, False)).available:
                        asyncio.create_task(self.switch_backend(backend_id))
                        return

            # Fallback to edge
            if self._backends.get("edge", TTSBackendInfo("", "", "", None, False, False)).available:
                asyncio.create_task(self.switch_backend("edge"))

        elif preference == "mlx":
            # Try kokoro first, then marvis
            for backend_id in ["mlx-kokoro", "mlx-marvis"]:
                if self._backends.get(backend_id, TTSBackendInfo("", "", "", None, False, False)).available:
                    asyncio.create_task(self.switch_backend(backend_id))
                    return

        elif preference == "qwen3":
            for backend_id in ["qwen3-cuda-0.6b", "qwen3-cuda-1.7b"]:
                if self._backends.get(backend_id, TTSBackendInfo("", "", "", None, False, False)).available:
                    asyncio.create_task(self.switch_backend(backend_id))
                    return

        elif preference == "edge":
            if self._backends.get("edge", TTSBackendInfo("", "", "", None, False, False)).available:
                asyncio.create_task(self.switch_backend("edge"))

    async def initialize_preferred(self, preference: str):
        """
        Initialize with preferred backend (async version).
        Preference can be: "mlx", "qwen3", "edge", "remote", or "auto"

        Tries backends in priority order, falling back if one fails to load.
        """
        import os
        mlx_model = os.environ.get("MLX_MODEL", "kokoro").lower()
        preference = preference.lower()

        # Build list of backends to try based on preference
        backends_to_try = []

        if preference == "auto":
            # Priority: mlx (Mac) -> qwen3 (CUDA) -> edge (cloud)
            if self._is_apple_silicon:
                # Try requested MLX model first
                backends_to_try.append(f"mlx-{mlx_model}")
                # Then other MLX models
                for bid in ["mlx-spark", "mlx-marvis", "mlx-kokoro"]:
                    if bid not in backends_to_try:
                        backends_to_try.append(bid)
            if self._has_cuda:
                backends_to_try.extend(["qwen3-cuda-0.6b", "qwen3-cuda-1.7b"])
            backends_to_try.append("edge")

        elif preference == "mlx":
            backends_to_try.append(f"mlx-{mlx_model}")
            for bid in ["mlx-spark", "mlx-marvis", "mlx-kokoro"]:
                if bid not in backends_to_try:
                    backends_to_try.append(bid)
            backends_to_try.append("edge")  # Fallback

        elif preference == "qwen3":
            backends_to_try.extend(["qwen3-cuda-0.6b", "qwen3-cuda-1.7b"])
            backends_to_try.append("edge")  # Fallback

        elif preference == "edge":
            backends_to_try.append("edge")

        elif preference == "remote":
            backends_to_try.append("remote")
            backends_to_try.append("edge")  # Fallback

        # Try each backend until one works
        for backend_id in backends_to_try:
            backend_info = self._backends.get(backend_id)
            if backend_info and backend_info.available:
                print(f"[TTS-MGR] Trying {backend_id}...")
                if await self.switch_backend(backend_id):
                    return True
                else:
                    print(f"[TTS-MGR] {backend_id} failed, trying next...")

        print("[TTS-MGR] All backends failed!")
        return False
