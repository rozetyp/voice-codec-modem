#!/usr/bin/env python3
"""
Aggressive Bitrate Probing: Find the limits.

Tests:
1. Ultra-short symbols (1ms, 2ms, 5ms, 10ms)
2. Higher arity (4-ary → 8-ary → 16-ary)
3. Resulting bitrates up to 48 kbps

Usage:
    python scripts/probe_aggressive.py --codec aac
    python scripts/probe_aggressive.py --codec opus --arity 8
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess

from voice_detector.audio_modem_aggressive import AudioModemAggressive
from voice_detector.codec_tester import CodecLoopbackTester


def run_aggressive_test(
    codec: str, test_data: bytes, symbol_ms: float, arity: int
) -> dict | None:
    """Run a single aggressive test."""
    try:
        modem = AudioModemAggressive(sample_rate=16000, symbol_duration_ms=symbol_ms, arity=arity)

        # Encode and modulate
        symbols = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_mfsk_signal(symbols, base_freq=200.0, add_voice_noise=True)

        # Write to WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
            sf.write(wav_path, audio, 16000, subtype="PCM_16")

        try:
            # Encode with codec
            tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_ms)
            encode_cmd = ["ffmpeg", "-i", str(wav_path), "-y", "-loglevel", "error"]

            # Get codec args from preset
            preset = tester.CODEC_PRESETS[codec]
            encode_args = preset.get("encode_args", [])
            ext = preset.get("ext", ".wav")

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_enc:
                enc_path = tmp_enc.name

            subprocess.run(encode_cmd + encode_args + [enc_path], check=True, capture_output=True)

            # Decode
            with open(enc_path, "rb") as f:
                encoded_bytes = f.read()

            decode_cmd = [
                "ffmpeg", "-i", enc_path, "-y", "-loglevel", "error",
                "-ar", "16000", "-acodec", "pcm_s16le"
            ]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_dec:
                dec_path = tmp_dec.name

            subprocess.run(decode_cmd + [dec_path], check=True, capture_output=True)
            audio_decoded, sr = sf.read(dec_path, dtype="float32")

            # Cleanup
            Path(enc_path).unlink(missing_ok=True)
            Path(dec_path).unlink(missing_ok=True)

            # Normalize
            audio_decoded = audio_decoded / (np.max(np.abs(audio_decoded)) + 1e-8) * 0.8

            # Demodulate
            symbols_recovered = modem.demodulate_mfsk(audio_decoded, base_freq=200.0)

            # Pad
            if len(symbols_recovered) < len(symbols):
                symbols_recovered = np.pad(
                    symbols_recovered, (0, len(symbols) - len(symbols_recovered))
                )
            else:
                symbols_recovered = symbols_recovered[: len(symbols)]

            # Convert back
            output_data = modem.symbols_to_binary(symbols_recovered)

            # Measure BER
            input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
            output_bits = np.unpackbits(
                np.frombuffer(output_data, dtype=np.uint8)
            ) if output_data else np.array([])

            max_len = max(len(input_bits), len(output_bits))
            input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
            output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))

            bit_errors = np.sum(input_bits != output_bits)
            ber = bit_errors / max_len if max_len > 0 else 1.0

            symbol_errors = np.sum(symbols != symbols_recovered)

            # Calculate bitrate
            bits_per_symbol = {4: 2, 8: 3, 16: 4}[arity]
            bitrate = bits_per_symbol * 1000 / symbol_ms

            return {
                "symbol_ms": symbol_ms,
                "arity": arity,
                "bits_per_symbol": bits_per_symbol,
                "bitrate_bps": bitrate,
                "ber": ber,
                "bit_errors": bit_errors,
                "total_bits": max_len,
                "symbol_errors": symbol_errors,
                "total_symbols": len(symbols),
            }
        finally:
            wav_path.unlink(missing_ok=True)

    except Exception as e:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggressive Bitrate Probing: Find the limit"
    )
    parser.add_argument("--codec", default="aac", help="Codec to test")
    parser.add_argument("--size", type=int, default=50, help="Test data size (bytes)")
    parser.add_argument("--arity", type=int, default=8, choices=[4, 8, 16],
                       help="Modulation arity (4/8/16)")
    parser.add_argument("--symbol-sweep", action="store_true",
                       help="Sweep symbol durations at fixed arity")
    parser.add_argument("--arity-sweep", action="store_true",
                       help="Sweep arity at fixed symbol duration")
    parser.add_argument("--full-probe", action="store_true",
                       help="Full probe: all arities × all symbol durations")

    args = parser.parse_args()

    test_data = np.random.bytes(args.size)

    print("=" * 100)
    print("AGGRESSIVE BITRATE PROBING: PUSHING TOWARD 50 KBPS")
    print("=" * 100)
    print(f"\nTest data: {args.size} bytes")
    print(f"Codec: {args.codec}\n")

    if args.full_probe or not (args.symbol_sweep or args.arity_sweep):
        # Full probe: test all combinations
        arities = [4, 8, 16]
        symbol_durations = [10, 5, 2, 1]

        print(
            f"{'Arity':<8} {'Symbols/ms':<12} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<10}"
        )
        print("-" * 100)

        results = []
        for arity in arities:
            for symbol_ms in symbol_durations:
                result = run_aggressive_test(args.codec, test_data, symbol_ms, arity)
                if result:
                    ber = result["ber"]
                    status = "✓" if ber <= 0.05 else "✗"
                    print(
                        f"{arity:<8} {1000/symbol_ms:<12.0f} {result['bitrate_bps']:<18.0f} "
                        f"{ber:<12.4f} {status:<10}"
                    )
                    results.append(result)
                else:
                    print(f"{arity:<8} {1000/symbol_ms:<12.0f} {'ERROR':<18}")

        # Find best achievable bitrate
        passing = [r for r in results if r["ber"] <= 0.05]
        if passing:
            best = max(passing, key=lambda x: x["bitrate_bps"])
            print("\n" + "=" * 100)
            print(f"🎯 BEST RESULT: {best['bitrate_bps']:.0f} bps ({best['arity']}-ary, {best['symbol_ms']}ms)")
            print(f"   BER: {best['ber']:.4f}, Symbol Errors: {best['symbol_errors']}/{best['total_symbols']}")
            print(f"   Approach target: {best['bitrate_bps']/50000*100:.1f}% of 50 kbps goal")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
