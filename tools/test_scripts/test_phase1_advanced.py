#!/usr/bin/env python3
"""
Phase 1 Advanced Testing: Bitrate Sweep + Noise Robustness.

Tests your modem at different symbol durations (bitrates) and SNR levels.
Finds the optimal operating point for your Vocal Modem.

Usage:
    python scripts/test_phase1_advanced.py --bitrate-sweep
    python scripts/test_phase1_advanced.py --noise-sweep --codec aac
    python scripts/test_phase1_advanced.py --full-matrix --codec opus --size 500

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from voice_detector.codec_tester import CodecLoopbackTester


def test_bitrate_sweep(
    codec: str,
    test_data: bytes,
    bitrates_ms: list[float] | None = None,
) -> dict:
    """
    Sweep across symbol durations to find bitrate/BER tradeoff.

    Args:
        codec: Codec to test.
        test_data: Test data (bytes).
        bitrates_ms: Symbol durations in milliseconds. Default: [100, 50, 20, 10].

    Returns:
        dict mapping bitrate_ms → result
    """
    if bitrates_ms is None:
        bitrates_ms = [100, 50, 20, 10]

    print(f"\n🎯 Bitrate Sweep: codec '{codec}'")
    print(f"   Test data: {len(test_data)} bytes")
    print(f"   Symbol durations: {bitrates_ms} ms\n")
    print(f"{'Duration (ms)':<15} {'Bitrate (bps)':<15} {'BER':<12} {'Symbols Err':<15} {'Status':<10}")
    print("-" * 70)

    results = {}

    for symbol_ms in bitrates_ms:
        bitrate_bps = (2 * 8) / (symbol_ms / 1000.0)  # 4-ary → 2 bits per symbol

        tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_ms)
        try:
            result = tester.run_loopback_test(test_data, base_freq=200.0, snr_db=None)

            ber = result["bit_error_rate"]
            ser = result["symbol_errors"]
            total_sym = result["total_symbols"]

            # Status: PASS if BER < 5%
            status = "✓ PASS" if ber <= 0.05 else "✗ FAIL"

            print(
                f"{symbol_ms:<15.1f} {bitrate_bps:<15.1f} {ber:<12.4f} "
                f"{ser}/{total_sym:<13} {status:<10}"
            )

            results[symbol_ms] = result
        except Exception as e:
            print(f"{symbol_ms:<15.1f} {'ERROR':<15} {str(e)[:40]:<40}")
            results[symbol_ms] = {"error": str(e)}

    return results


def test_noise_sweep(
    codec: str,
    test_data: bytes,
    snr_db_list: list[float] | None = None,
    symbol_duration_ms: float = 100.0,
) -> dict:
    """
    Sweep across SNR levels to test noise robustness.

    Args:
        codec: Codec to test.
        test_data: Test data (bytes).
        snr_db_list: SNR levels in dB. Default: [50, 30, 20, 10, 5, 0].
        symbol_duration_ms: Symbol duration (default 100ms).

    Returns:
        dict mapping snr_db → result
    """
    if snr_db_list is None:
        snr_db_list = [50, 30, 20, 10, 5, 0]

    print(f"\n📊 Noise Sweep: codec '{codec}', symbol duration {symbol_duration_ms} ms")
    print(f"   Test data: {len(test_data)} bytes\n")
    print(f"{'SNR (dB)':<12} {'BER':<12} {'Symbols Err':<15} {'Status':<10}")
    print("-" * 50)

    results = {}
    tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_duration_ms)

    for snr_db in snr_db_list:
        try:
            result = tester.run_loopback_test(test_data, base_freq=200.0, snr_db=snr_db)

            ber = result["bit_error_rate"]
            ser = result["symbol_errors"]
            total_sym = result["total_symbols"]

            status = "✓ PASS" if ber <= 0.05 else "✗ FAIL"

            print(
                f"{snr_db:<12.1f} {ber:<12.4f} {ser}/{total_sym:<13} {status:<10}"
            )

            results[snr_db] = result
        except Exception as e:
            print(f"{snr_db:<12.1f} {'ERROR':<40}")
            results[snr_db] = {"error": str(e)}

    return results


def test_full_matrix(
    codec: str,
    test_data: bytes,
    bitrates_ms: list[float] | None = None,
    snr_db_list: list[float] | None = None,
) -> dict:
    """
    Full matrix test: all combinations of bitrate and SNR.

    Returns:
        dict mapping (bitrate, snr) → result
    """
    if bitrates_ms is None:
        bitrates_ms = [100, 50, 20]
    if snr_db_list is None:
        snr_db_list = [30, 20, 10]

    print(f"\n🧬 Full Matrix Test: codec '{codec}'")
    print(f"   Test data: {len(test_data)} bytes")
    print(f"   Symbol durations: {bitrates_ms} ms")
    print(f"   SNR levels: {snr_db_list} dB\n")

    results = {}

    print(f"{'Duration':<12} {' | '.join([f'SNR={s:>3.0f}dB' for s in snr_db_list])}")
    print("-" * 70)

    for symbol_ms in bitrates_ms:
        row_label = f"{symbol_ms:.0f}ms"
        row_results = []

        for snr_db in snr_db_list:
            try:
                tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_ms)
                result = tester.run_loopback_test(test_data, base_freq=200.0, snr_db=snr_db)

                ber = result["bit_error_rate"]
                status = "✓" if ber <= 0.05 else "✗"
                row_results.append(f"{ber:.3f}{status}")

                results[(symbol_ms, snr_db)] = result
            except Exception as e:
                row_results.append("ERROR")
                results[(symbol_ms, snr_db)] = {"error": str(e)}

        print(f"{row_label:<12} {' | '.join([f'{r:>8}' for r in row_results])}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Advanced: Bitrate & Noise Analysis")
    parser.add_argument("--codec", default="aac", help="Codec to test (default: aac)")
    parser.add_argument("--size", type=int, default=100, help="Test data size in bytes (default: 100)")
    parser.add_argument(
        "--bitrate-sweep",
        action="store_true",
        help="Test across symbol durations (100ms → 10ms)",
    )
    parser.add_argument(
        "--noise-sweep",
        action="store_true",
        help="Test across SNR levels (clean → 0dB noise)",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Full matrix: all bitrate × SNR combinations",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results as JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Save results as CSV (for matrix only)",
    )

    args = parser.parse_args()

    # Prepare test data
    test_data = np.random.bytes(args.size)

    print("=" * 70)
    print("PHASE 1 ADVANCED: BITRATE & NOISE ANALYSIS")
    print("=" * 70)

    results = None

    if args.full_matrix:
        results = test_full_matrix(args.codec, test_data)
    elif args.noise_sweep:
        results = test_noise_sweep(args.codec, test_data)
    elif args.bitrate_sweep:
        results = test_bitrate_sweep(args.codec, test_data)
    else:
        results = test_bitrate_sweep(args.codec, test_data)

    # Save results
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_data = {}
        if isinstance(results, dict):
            for key, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    json_data[str(key)] = {
                        k: (v.hex() if isinstance(v, bytes) else (v.tolist() if isinstance(v, np.ndarray) else v))
                        for k, v in result.items()
                    }
                else:
                    json_data[str(key)] = result

        with open(args.output_json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n💾 JSON results saved to {args.output_json}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
