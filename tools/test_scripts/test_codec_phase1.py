#!/usr/bin/env python3
"""
Phase 1 Test Runner: Codec Robustness Loopback.

Usage:
    python scripts/test_codec_phase1.py --codec pcm --test-data "Hello, Vocal Modem!"
    python scripts/test_codec_phase1.py --codec aac --freq-range 150-400 --freq-step 50
    python scripts/test_codec_phase1.py --codec amr-wb --size 1000

Tests your encoded data against voice codecs at varying bitrates and frequencies.
Measures BER (Bit Error Rate). Goal: BER < 5%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voice_detector.codec_tester import CodecLoopbackTester


def format_result(result: dict) -> str:
    """Pretty-print a test result."""
    return (
        f"  Codec: {result['codec']:<8} | "
        f"BER: {result['bit_error_rate']:.4f} ({result['bit_errors']}/{result['total_bits']} bits) | "
        f"SER: {result['symbol_errors']}/{result['total_symbols']} symbols | "
        f"Freq: {result['base_freq']:.0f} Hz"
    )


def test_single(codec: str, test_data: bytes, base_freq: float = 200.0) -> dict:
    """Run a single loopback test."""
    tester = CodecLoopbackTester(codec=codec, sample_rate=16000)
    result = tester.run_loopback_test(test_data, base_freq=base_freq)
    return result


def test_codec_sweep(
    codec: str,
    test_data: bytes,
    freq_min: float = 100,
    freq_max: float = 400,
    freq_step: float = 50,
) -> list[dict]:
    """Test across a range of base frequencies."""
    results = []
    freqs = np.arange(freq_min, freq_max + freq_step, freq_step)

    print(f"\n🔬 Testing codec '{codec}' across {len(freqs)} frequencies...")
    print(f"   Frequency range: {freq_min}–{freq_max} Hz, step: {freq_step} Hz")
    print(f"   Test data size: {len(test_data)} bytes\n")

    for freq in freqs:
        result = test_single(codec, test_data, base_freq=freq)
        results.append(result)
        print(format_result(result))

    # Summary
    bers = [r["bit_error_rate"] for r in results]
    best_freq = freqs[int(np.argmin(bers))]
    best_ber = min(bers)
    print(f"\n✓ Best BER: {best_ber:.4f} at {best_freq:.0f} Hz")
    if best_ber <= 0.05:
        print("  ✓ BER < 5% threshold PASSED ✓")
    else:
        print(f"  ✗ BER FAILED: {best_ber:.4f} > 0.05 target")
        print("  → Recommendation: Spread frequencies further apart or reduce bitrate")

    return results


def test_all_codecs(codec_list: list[str], test_data: bytes) -> dict:
    """Compare all given codecs at a fixed frequency."""
    print(f"\n🔬 Comparing codecs: {', '.join(codec_list)}")
    print(f"   Test data size: {len(test_data)} bytes\n")

    all_results = {}
    for codec in codec_list:
        try:
            result = test_single(codec, test_data, base_freq=200.0)
            all_results[codec] = result
            print(format_result(result))
        except Exception as e:
            print(f"  {codec:<8}: ERROR — {e}")
            all_results[codec] = None

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: test codec robustness via loopback test (modulate → encode → decode → demodulate)."
    )
    parser.add_argument(
        "--codec",
        default="pcm",
        help="Codec to test (pcm, aac, opus, amr-wb). Default: pcm (baseline).",
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="Test data as string. Default: random bytes.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Size of random test data (bytes) if --test-data not provided. Default: 100.",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=150,
        help="Min frequency sweep (Hz). Default: 150.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=400,
        help="Max frequency sweep (Hz). Default: 400.",
    )
    parser.add_argument(
        "--freq-step",
        type=float,
        default=50,
        help="Frequency step (Hz). Default: 50.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep frequencies instead of testing at fixed freq.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all codecs (pcm, aac, opus, amr-wb).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results as JSON.",
    )

    args = parser.parse_args()

    # Prepare test data
    if args.test_data:
        test_data = args.test_data.encode("utf-8")
    else:
        test_data = np.random.bytes(args.size)

    print("=" * 70)
    print("PHASE 1: CODEC ROBUSTNESS LOOPBACK TEST")
    print("=" * 70)

    results = None

    if args.compare_all:
        results = test_all_codecs(["pcm", "aac", "opus", "amr-nb"], test_data)
    elif args.sweep:
        results = test_codec_sweep(
            args.codec,
            test_data,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            freq_step=args.freq_step,
        )
    else:
        result = test_single(args.codec, test_data, base_freq=args.freq_min)
        print(f"\n🔬 Single test: codec '{args.codec}'")
        print(format_result(result))
        results = result

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Convert to JSON-serializable format
        if isinstance(results, list):
            json_data = [
                {k: (v.hex() if isinstance(v, bytes) else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                 for k, v in r.items()} for r in results
            ]
        elif isinstance(results, dict):
            if all(isinstance(v, dict) for v in results.values()):
                json_data = {
                    k: {kk: (vv.hex() if isinstance(vv, bytes) else (vv.tolist() if isinstance(vv, np.ndarray) else vv)) 
                        for kk, vv in v.items()}
                    for k, v in results.items()
                    if v is not None
                }
            else:
                json_data = {k: (v.hex() if isinstance(v, bytes) else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                            for k, v in results.items()}
        else:
            json_data = results

        with open(args.output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
