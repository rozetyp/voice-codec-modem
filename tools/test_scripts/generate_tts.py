"""Generate a single ElevenLabs TTS reading of a text passage.

Usage:
    python scripts/generate_tts.py --text passage.txt --out data/raw/synth_readaloud_01.wav

Reads $ELEVENLABS_API_KEY and $ELEVENLABS_VOICE_ID from .env.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path, help="path to a text file to be read aloud")
    parser.add_argument("--out", required=True, type=Path, help="output WAV path")
    parser.add_argument("--voice-id", default=os.environ.get("ELEVENLABS_VOICE_ID"))
    parser.add_argument("--model-id", default="eleven_multilingual_v2")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY not set — copy .env.example to .env and fill in.")
    if not args.voice_id:
        raise SystemExit("no --voice-id and ELEVENLABS_VOICE_ID not set")

    text = args.text.read_text().strip()
    if len(text) < 200:
        print(f"WARNING: passage is only {len(text)} chars — aim for ≥5 min of reading (~4000 chars).")

    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        voice_id=args.voice_id,
        text=text,
        model_id=args.model_id,
        output_format="pcm_16000",
        voice_settings=VoiceSettings(
            stability=args.stability,
            similarity_boost=args.similarity_boost,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # PCM 16kHz mono — wrap in WAV container
    import wave

    pcm = b"".join(audio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)
    print(f"wrote {args.out}  ({len(pcm)/2/16000:.1f}s)")


if __name__ == "__main__":
    main()
