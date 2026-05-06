from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


def transcribe(
    audio: np.ndarray | Path | str,
    model_name: str = "large-v3",
    language: str | None = "en",
    vad_filter: bool = True,
) -> list[Segment]:
    """Run faster-whisper with word timestamps. Returns segments (whisper's own split).

    For short files this loads the model every call; for batches use `load_model`
    and `transcribe_with` below.
    """
    model = load_model(model_name)
    return transcribe_with(model, audio, language=language, vad_filter=vad_filter)


def load_model(model_name: str = "large-v3"):
    from faster_whisper import WhisperModel

    # int8 runs cleanly on Apple Silicon CPU; float16 needs CUDA
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_with(
    model,
    audio: np.ndarray | Path | str,
    language: str | None = "en",
    vad_filter: bool = True,
) -> list[Segment]:
    audio_arg = str(audio) if isinstance(audio, (str, Path)) else audio
    segments_iter, _info = model.transcribe(
        audio_arg,
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    out: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(text=w.word.strip(), start=float(w.start), end=float(w.end))
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        out.append(Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip(), words=words))
    return out
