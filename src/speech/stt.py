"""Local speech-to-text with faster-whisper on CPU (int8). Audio never leaves the machine."""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache

from src import config

logger = logging.getLogger(__name__)

WHISPER_DIR = config.PROJECT_ROOT / ".hf_cache" / "whisper"
# Biases decoding towards the vocabulary of code questions.
INITIAL_PROMPT = (
    "Questions about a JavaScript codebase. Where is the deeplink used? Which files call "
    "setValue before emit? Who calls purgeStates? Where is the event emitted? Bluetooth, MQTT, "
    "Zigbee, intent, API endpoint, function."
)


@lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        config.WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=config.TORCH_THREADS,
        download_root=str(WHISPER_DIR),
    )


def transcribe(audio: bytes) -> tuple[str, float]:
    """Transcribe WAV/AIFF/MP3/WebM bytes. Returns (text, elapsed_ms)."""
    t0 = time.perf_counter()
    segments, _info = get_whisper().transcribe(
        io.BytesIO(audio),
        language="en",
        beam_size=1,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, round((time.perf_counter() - t0) * 1000, 1)
