#!/usr/bin/env python3
"""
Phase 2: Opus Bitrate Sweep
Find maximum sustainable bitrate with Opus codec.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_detector.music_mode_modem import MusicModeModem


def test_opus_loopback(
    audio: np.ndarray,
    modem: MusicModeModem,
    test_data: bytes,
    bitrate_kb: int = 32,
) -> dict:
    """Quick Opus codec loopback test."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save original
        orig_file = tmpdir / "original.wav"
        sf.write(orig_file, audio, 16000)
        
        # Encode to Opus
        encoded_file = tmpdir / "encoded.opus"
        cmd = [
            "ffmpeg", "-i", str(orig_file), "-c:a", "libopus",
            "-b:a", f"{bitrate_kb}k", "-v", "0", "-y", str(encoded_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            return {"error": str(e), "ber": 100}
        
        # Decode back
        decoded_file = tmpdir / "decoded.wav"
        cmd = [
            "ffmpeg", "-i", str(encoded_file), "-ar", "16000", "-ac", "1",
            "-v", "0", "-y", str(decoded_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            return {"error": str(e), "ber": 100}
        
        # Load decoded
        try:
            decoded_audio, sr = sf.read(decoded_file)
        except Exception as e:
            return {"error": str(e), "ber": 100}
        
        # Demodulate
        try:
            symbols = modem.demodulate_chirp(decoded_audio)
            recovered_data = modem.symbols_to_binary(symbols)
        except Exception as e:
            return {"error": str(e), "ber": 100}
        
        # Calculate BER
        bits_total = len(test_data) * 8
        bits_error = 0
        
        for i, orig_byte in enumerate(test_data):
            if i < len(recovered_data):
                recov_byte = recovered_data[i]
                bits_error += bin(orig_byte ^ recov_byte).count('1')
            else:
                bits_error += 8
        
        ber = (bits_error / bits_total * 100) if bits_total > 0 else 100
        
        return {"ber": ber, "bits_error": bits_error}


def main():
    print("\n" + "=" * 100)
    print("PHASE 2: OPUS BITRATE SWEEP - Find Maximum Sustainable Rate")
    print("=" * 100)
    
    # Use substantial test data for realistic bitrates
    test_data = b"Phase2OmusMaxBitrateTest" * 10  # 240 bytes
    
    symbol_durations = [
        (5.0, "5ms (safest)"),
        (3.0, "3ms (safe)"),
        (2.0, "2ms (good)"),
        (1.5, "1.5ms (risky)"),
        (1.0, "1ms (aggressive)"),
        (0.75, "0.75ms (very aggressive)"),
        (0.5, "0.5ms (extreme)"),
    ]
    
    print(f"\nTest data: {len(test_data)} bytes")
    print(f"\n{'Symbol Duration':<20} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<15} {'Verdict'}")
    print("-" * 100)
    
    best_bitrate = 0
    best_config = None
    
    for sym_dur, label in symbol_durations:
        modem = MusicModeModem(
            symbol_duration_ms=sym_dur,
            chirp_overlap=0.25
        )
        
        symbols = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_music_mode_signal(symbols, add_music_floor=True)
        bitrate = (len(test_data) * 8) / (len(audio) / 16000)
        
        result = test_opus_loopback(audio, modem, test_data)
        ber = result.get("ber", 100)
        
        if ber < 5:
            status = "✓ PASS"
            verdict = "Safe"
            if bitrate > best_bitrate:
                best_bitrate = bitrate
                best_config = (sym_dur, label, bitrate, ber)
        elif ber < 10:
            status = "⚠ MARGINAL"
            verdict = "Risky"
        else:
            status = "✗ FAIL"
            verdict = "Too fast"
        
        print(f"{label:<20} {bitrate:<18.0f} {ber:<12.2f}% {status:<15} {verdict}")
    
    print("\n" + "=" * 100)
    if best_config:
        sym_dur, label, bitrate, ber = best_config
        print(f"PHASE 2 CHAMPION (with Opus):")
        print(f"  Symbol duration: {label}")
        print(f"  Bitrate: {bitrate:.0f} bps")
        print(f"  BER: {ber:.2f}%")
        print(f"  Status: ✓ EXCELLENT")
        print(f"\n  This is {bitrate/533*100:.1f}% of Phase 1 baseline (533 bps at 5ms)")
        print(f"  Ready for Phase 3: Hardware testing on real VoLTE")
    else:
        print("No safe configuration found!")
    print("=" * 100)


if __name__ == "__main__":
    main()
