"""Extract per-turn features from every WAV in data/raw/ and save to data/features/.

Convention:
  data/raw/human_*.wav   -> mono, single-speaker recording of a human
  data/raw/synth_*.wav   -> mono, single-speaker synthetic recording

  data/raw/call_*.wav    -> stereo, channel 0 = interviewer, channel 1 = responder.
                             Processed as two separate tracks:
                               call_foo_ch0 (interviewer, ignored for now)
                               call_foo_ch1 (responder - scored)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import soundfile as sf

from voice_detector.features import extract_turn_features, segments_to_turns
from voice_detector.transcribe import load_model, transcribe_with


def process_one(audio_path: Path, model, out_dir: Path) -> Path | None:
    out = out_dir / f"{audio_path.stem}.parquet"
    if out.exists():
        print(f"  SKIP {audio_path.name} — features already exist at {out.name}")
        return out
    print(f"  transcribing {audio_path.name}")
    segments = transcribe_with(model, audio_path)
    turns = segments_to_turns(segments)
    if len(turns) < 10:
        print(f"  SKIP {audio_path.name} — only {len(turns)} turns")
        return None
    print(f"  extracting features ({len(turns)} turns)")
    df = extract_turn_features(audio_path, turns)
    df.write_parquet(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    args.features_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.raw_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no WAVs in {args.raw_dir}")

    print(f"loading whisper {args.model} ...")
    model = load_model(args.model)

    produced: list[Path] = []
    for wav in wavs:
        # auto-split stereo 'call_*' files into per-channel mono
        info = sf.info(str(wav))
        if info.channels == 2 and wav.name.startswith("call_"):
            from voice_detector.audio import split_stereo

            split_out = args.raw_dir / "_split"
            ch0, ch1 = split_stereo(wav, split_out)
            # Score only channel 1 (responder) — flip if your Vapi recording is reversed
            out = process_one(ch1, model, args.features_dir)
            if out:
                produced.append(out)
        else:
            out = process_one(wav, model, args.features_dir)
            if out:
                produced.append(out)

    print(f"\nproduced {len(produced)} feature files in {args.features_dir}")


if __name__ == "__main__":
    main()
