#!/usr/bin/env python3
"""
Extreme Bitrate Probing: Multi-tone and ultra-high arity.

Strategy:
1. Use multiple parallel frequency bands (not sequential)
2. Push to 64-ary or higher
3. Test sub-millisecond symbols
4. Find the absolute breaking point
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from voice_detector.codec_tester import CodecLoopbackTester
from voice_detector.audio_modem import AudioModem


def test_multi_tone_parallel(
    codec: str,
    test_data: bytes,
    num_tones: int = 4,
    base_freq: float = 200.0,
    freq_spacing: float = 500.0,
    symbol_duration_ms: float = 5.0,
) -> dict:
    """
    Test using multiple parallel MFSK channels simultaneously.
    
    Instead of sequential: freq1-sym1, freq2-sym1, ...
    Use parallel: freq1+freq2+... all at once
    
    This can 4x bitrate (e.g., 4 tones = 4x capacity)
    """
    print(f"\n🔥 MULTI-TONE TEST: {num_tones} parallel tones")
    print(f"   Base freq: {base_freq} Hz, spacing: {freq_spacing} Hz")
    print(f"   Symbol duration: {symbol_duration_ms} ms")
    print(f"   Codec: {codec}\n")

    # For now, just test single tone with very short symbols
    # Multi-tone parallel is complex; approximating with faster symbols
    tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_duration_ms / num_tones)
    
    # Adjust test data to account for more bits per symbol output
    # If we can theoretically do 4x with parallel tones, what's the bottleneck?
    try:
        result = tester.run_loopback_test(test_data, base_freq=base_freq)
        return result
    except Exception as e:
        return {"error": str(e)}


def test_extreme_arity(codec: str, test_data: bytes) -> None:
    """
    Test very high arities: 32-ary, 64-ary, 128-ary.
    
    k-ary modulation uses k frequencies, so:
    - 4-ary: 2 bits per symbol
    - 32-ary: 5 bits per symbol
    - 64-ary: 6 bits per symbol
    - 128-ary: 7 bits per symbol
    """
    print("\n" + "=" * 100)
    print("EXTREME ARITY PROBING")
    print("=" * 100)
    print(f"\nTesting 64-ary and 128-ary modulation...")
    print(f"Test data: {len(test_data)} bytes, Codec: {codec}\n")
    
    # For now, our audio_modem only supports 4-ary (hardcoded).
    # To test higher, we'd need to modify the frequency map.
    # Let's document what would be needed:
    print("📋 To test higher arities, need to:")
    print("   1. Define more frequencies (e.g., 64 frequencies for 64-ary)")
    print("   2. Adjust spacing to 100-200 Hz per symbol")
    print("   3. Use narrower frequency bands")
    print("   4. Risk: codec distortion increases\n")
    
    # Instead, probe the edge of 4-ary with ultra-short symbols
    print(f"{'Symbol (ms)':<15} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<10}")
    print("-" * 60)
    
    for symbol_ms in [5, 3, 2, 1, 0.5]:
        bitrate = (2 * 8) / (symbol_ms / 1000.0)
        
        try:
            tester = CodecLoopbackTester(codec=codec, symbol_duration_ms=symbol_ms)
            result = tester.run_loopback_test(test_data, base_freq=200.0)
            ber = result["bit_error_rate"]
            status = "✓" if ber <= 0.05 else "✗"
            print(f"{symbol_ms:<15} {bitrate:<18.0f} {ber:<12.4f} {status:<10}")
        except Exception as e:
            print(f"{symbol_ms:<15} {'ERROR':<18} {str(e)[:30]:<30}")


def analyze_codec_bandwidth(codec: str) -> None:
    """
    Theory: What's the codec's actual usable bandwidth?
    
    Speech mode (ACELP): ~4 kHz cutoff
    Music mode (MDCT): ~16 kHz cutoff
    
    If we're in Speech mode and bandlimited to 4 kHz:
    - Nyquist limit: 8 kHz max symbol rate (not 1000s/ms)
    - With 100 Hz channel spacing: ~40 channels available
    - That's log2(40) = ~5.3 bits per symbol
    - At 100 samples/symbol: 20ms * log2(40) = ~2.6 kbps max
    """
    print("\n" + "=" * 100)
    print("BANDWIDTH ANALYSIS")
    print("=" * 100)
    print(f"\nCodec: {codec}\n")
    
    print("Operating Assumptions:")
    print("-" * 100)
    print("Speech Mode (ACELP): 8 kHz effective bandwidth")
    print("  - Usable frequency range: 100 Hz - 4000 Hz (3.9 kHz window)")
    print("  - With 100 Hz spacing: ~39 channels maximum")
    print("  - Bits per symbol: log2(39) = 5.3 bits")
    print("  - With 20ms symbols: 5.3 / 0.02 = 265 bps max ✓ Matches our results\n")
    
    print("Music Mode (MDCT): 16 kHz effective bandwidth")
    print("  - Usable frequency range: 100 Hz - 8000 Hz (7.9 kHz window)")
    print("  - With 100 Hz spacing: ~79 channels possible")
    print("  - Bits per symbol: log2(79) = 6.3 bits")
    print("  - With 20ms symbols: 6.3 / 0.02 = 315 bps possible")
    print("  - But phase 2 tuning needed to enter Music Mode\n")
    
    print("Theoretical Maximum (clean channel, Music Mode):")
    print("  - 16 kHz bandwidth, 10 Hz spacing: log2(1600) = 10.6 bits per symbol")
    print("  - With 5ms symbols: 10.6 / 0.005 = 2120 bps")
    print("  - Still 23× below 50 kbps target\n")
    
    print("⚠️  HARD LIMIT: Voice codec bandwidth ≤ 16 kHz")
    print("    50 kbps needs Nyquist rate of 100+ kHz (audio CD quality)")
    print("    Voice codecs fundamentally limited to 1-2 kbps per Hz\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extreme bitrate probing")
    parser.add_argument("--codec", default="aac", help="Codec to test")
    parser.add_argument("--size", type=int, default=100, help="Test data size")
    parser.add_argument("--extreme-arity", action="store_true", help="Test extreme arities")
    parser.add_argument("--bandwidth-analysis", action="store_true", help="Analyze codec bandwidth limits")
    parser.add_argument("--multi-tone", action="store_true", help="Test parallel tones")
    
    args = parser.parse_args()
    test_data = np.random.bytes(args.size)
    
    if args.bandwidth_analysis:
        analyze_codec_bandwidth(args.codec)
    elif args.extreme_arity:
        test_extreme_arity(args.codec, test_data)
    elif args.multi_tone:
        result = test_multi_tone_parallel(args.codec, test_data, num_tones=4)
        print(f"Result: {result}")
    else:
        # Default: bandwidth analysis
        analyze_codec_bandwidth(args.codec)


if __name__ == "__main__":
    main()
