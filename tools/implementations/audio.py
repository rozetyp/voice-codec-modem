from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: Path | str, target_sr: int = 16000, channel: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV, optionally select a channel, return (samples, sr) as float32 mono.

    - If file is stereo and `channel` is given, return that channel.
    - If file is stereo and `channel` is None, average to mono.
    - Resamples to `target_sr` if needed.
    """
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if channel is not None:
        if channel >= data.shape[1]:
            raise ValueError(f"requested channel {channel} but file has {data.shape[1]}")
        mono = data[:, channel]
    else:
        mono = data.mean(axis=1)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    return mono, sr


def split_stereo(path: Path | str, out_dir: Path | str) -> tuple[Path, Path]:
    """Split a stereo WAV into two mono WAVs — channel 0 and channel 1.

    Returns (ch0_path, ch1_path). Assumes Vapi convention:
      channel 0 = agent/interviewer, channel 1 = responder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] != 2:
        raise ValueError(f"expected stereo, got {data.shape[1]} channel(s)")
    stem = Path(path).stem
    ch0 = out_dir / f"{stem}_ch0.wav"
    ch1 = out_dir / f"{stem}_ch1.wav"
    sf.write(ch0, data[:, 0], sr)
    sf.write(ch1, data[:, 1], sr)
    return ch0, ch1
