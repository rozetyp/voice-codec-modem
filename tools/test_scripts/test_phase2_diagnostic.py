#!/usr/bin/env python3
"""
Phase 2: Diagnostic - Test Demodulation Quality
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_detector.music_mode_modem import MusicModeModem


def test_demodulation_no_codec():
    """Test demodulation WITHOUT codec (should be perfect)."""
    print("\n" + "=" * 90)
    print("DIAGNOSTIC: Demodulation WITHOUT Codec")
    print("=" * 90)
    
    test_data = b"TestPhase2"
    modem = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    
    # Encode
    symbols = modem.encode_binary_to_symbols(test_data)
    print(f"\nOriginal data: {test_data}")
    print(f"Encoded symbols (first 20): {symbols[:20]}")
    
    # Generate with music floor
    audio_with_floor = modem.generate_music_mode_signal(symbols, add_music_floor=True)
    
    # Generate without music floor
    audio_no_floor = modem.generate_music_mode_signal(symbols, add_music_floor=False)
    
    # Demodulate both
    print(f"\n--- Demodulation Results ---")
    
    for label, audio in [("Without music floor", audio_no_floor), ("With music floor", audio_with_floor)]:
        recovered_symbols = modem.demodulate_chirp(audio)
        recovered_data = modem.symbols_to_binary(recovered_symbols)
        
        # Calculate symbol error rate
        sym_errors = np.sum(symbols != recovered_symbols[:len(symbols)])
        ser = (sym_errors / len(symbols)) * 100
        
        # Calculate bit error rate
        bits_total = len(test_data) * 8
        bits_error = 0
        for orig, recov in zip(test_data, recovered_data[:len(test_data)]):
            bits_error += bin(orig ^ recov).count('1')
        ber = (bits_error / bits_total) * 100
        
        print(f"\n{label}:")
        print(f"  Recovered data: {recovered_data[:len(test_data)]}")
        print(f"  Symbol errors: {sym_errors}/{len(symbols)} ({ser:.1f}%)")
        print(f"  Bit errors: {bits_error}/{bits_total} ({ber:.1f}%)")
        print(f"  Status: {'✓ PERFECT' if ber == 0 else '✗ DEGRADED'}")


def test_with_noise():
    """Test with AWGN at various SNR levels."""
    print("\n" + "=" * 90)
    print("DIAGNOSTIC: Demodulation with AWGN Noise")
    print("=" * 90)
    
    test_data = b"TestPhase2" * 2
    modem = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    
    symbols = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_music_mode_signal(symbols, add_music_floor=True)
    
    snr_levels = [50, 30, 20, 10, 5, 0, -5]
    
    print(f"\n{'SNR (dB)':<12} {'Symbol Errors':<20} {'Bit Errors':<20} {'Status'}")
    print("-" * 90)
    
    for snr_db in snr_levels:
        # Add noise
        noisy_audio = modem.add_awgn(audio, snr_db=snr_db)
        
        # Demodulate
        recovered_symbols = modem.demodulate_chirp(noisy_audio)
        recovered_data = modem.symbols_to_binary(recovered_symbols)
        
        # Calculate errors
        sym_errors = np.sum(symbols != recovered_symbols[:len(symbols)])
        ser = (sym_errors / len(symbols)) * 100
        
        bits_total = len(test_data) * 8
        bits_error = 0
        for orig, recov in zip(test_data, recovered_data[:len(test_data)]):
            bits_error += bin(orig ^ recov).count('1')
        ber = (bits_error / bits_total) * 100
        
        status = "✓" if ber < 5 else "⚠" if ber < 20 else "✗"
        print(f"{snr_db:<12} {sym_errors:<20} {bits_error:<20} {status} {ber:.1f}% BER")


def test_symbol_detection():
    """Debug: test individual symbol detection."""
    print("\n" + "=" * 90)
    print("DIAGNOSTIC: Individual Symbol Detection")
    print("=" * 90)
    
    modem = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    
    # Generate each symbol type individually
    print(f"\n{'Symbol':<10} {'Freq0 (Hz)':<15} {'Freq1 (Hz)':<15} {'Detected':<10} {'Status'}")
    print("-" * 90)
    
    for sym in range(4):
        symbols = np.array([sym], dtype=np.uint8)
        audio = modem.generate_music_mode_signal(symbols, add_music_floor=False)
        
        recovered = modem.demodulate_chirp(audio)
        detected_sym = recovered[0] if len(recovered) > 0 else 0
        
        f0 = modem.base_freq + sym * modem.freq_spacing
        f1 = f0 + modem.freq_spacing
        
        status = "✓" if detected_sym == sym else "✗"
        print(f"{sym:<10} {f0:<15.0f} {f1:<15.0f} {detected_sym:<10} {status}")


if __name__ == "__main__":
    test_demodulation_no_codec()
    test_with_noise()
    test_symbol_detection()
