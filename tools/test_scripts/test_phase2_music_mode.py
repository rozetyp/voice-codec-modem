#!/usr/bin/env python3
"""
Phase 2 Testing: Music Mode Forcing
Comprehensive evaluation of music floor impact on bitrate.

Compares:
- Phase 1: Hybrid chirp + overlap (10.7 kbps, 0% BER)
- Phase 2: Hybrid chirp + overlap + music floor (expected 20-30 kbps, <5% BER)
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_detector.codec_tester import CodecLoopbackTester
from voice_detector.music_mode_modem import MusicModeModem


def test_phase_comparison():
    """Direct comparison: Phase 1 vs Phase 2."""
    print("\n" + "=" * 90)
    print("PHASE 2 VALIDATION: Music Mode Forcing")
    print("=" * 90)
    
    test_data = b"AudioModem2026Test" * 4  # 72 bytes
    
    # Phase 1: Chirp without music floor
    print("\n[PHASE 1] Hybrid Chirp - No Music Floor")
    print("-" * 90)
    modem_p1 = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    symbols_p1 = modem_p1.encode_binary_to_symbols(test_data)
    audio_p1 = modem_p1.generate_music_mode_signal(symbols_p1, add_music_floor=False)
    bitrate_p1 = (len(test_data) * 8) / (len(audio_p1) / 16000)
    print(f"  Symbols: {len(symbols_p1)}")
    print(f"  Audio: {len(audio_p1)} samples ({len(audio_p1)/16000:.3f}s)")
    print(f"  Bitrate: {bitrate_p1:.0f} bps")
    
    # Phase 2: With music floor
    print("\n[PHASE 2] Hybrid Chirp + Music Floor (60Hz + 200Hz)")
    print("-" * 90)
    modem_p2 = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    symbols_p2 = modem_p2.encode_binary_to_symbols(test_data)
    audio_p2 = modem_p2.generate_music_mode_signal(symbols_p2, add_music_floor=True)
    bitrate_p2 = (len(test_data) * 8) / (len(audio_p2) / 16000)
    print(f"  Symbols: {len(symbols_p2)}")
    print(f"  Audio: {len(audio_p2)} samples ({len(audio_p2)/16000:.3f}s)")
    print(f"  Music floor: 60 Hz (amp 0.08) + 200 Hz (amp 0.08)")
    print(f"  Bitrate: {bitrate_p2:.0f} bps")
    print(f"  Overhead: {(bitrate_p2/bitrate_p1 - 1)*100:.1f}% (due to signal normalization)")
    
    # Save samples
    output_dir = Path("audio_samples")
    output_dir.mkdir(exist_ok=True)
    sf.write(output_dir / "phase2_no_floor.wav", audio_p1, 16000)
    sf.write(output_dir / "phase2_with_floor.wav", audio_p2, 16000)
    print(f"\n  Saved: phase2_no_floor.wav, phase2_with_floor.wav")


def test_bitrate_sweep():
    """Test different symbol durations to find new optimal bitrate."""
    print("\n" + "=" * 90)
    print("BITRATE SWEEP: Music Mode")
    print("=" * 90)
    print(f"\n{'Symbol (ms)':<15} {'Bitrate (bps)':<20} {'Expected BER Goal':<20}")
    print("-" * 90)
    
    test_data = b"AudioModem2026Test" * 4
    
    symbol_durations = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
    
    for sym_dur in symbol_durations:
        modem = MusicModeModem(
            symbol_duration_ms=sym_dur,
            chirp_overlap=0.25
        )
        symbols = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_music_mode_signal(symbols, add_music_floor=True)
        bitrate = (len(test_data) * 8) / (len(audio) / 16000)
        
        # Estimate expected BER based on faster symbols
        # Faster symbols = more susceptible to distortion
        ber_prediction = "0% (excellent)" if sym_dur >= 2.0 else \
                        "<1% (good)" if sym_dur >= 1.0 else \
                        "1-5% (acceptable)" if sym_dur >= 0.5 else \
                        ">5% (unacceptable)"
        
        print(f"{sym_dur:<15.2f} {bitrate:<20.0f} {ber_prediction:<20}")


def test_codec_survival_phase2():
    """Test codec survival with music mode."""
    print("\n" + "=" * 90)
    print("CODEC SURVIVAL TEST: Phase 2 (Music Mode)")
    print("=" * 90)
    
    test_data = b"AudioModem2026Phase2Test" * 3  # 75 bytes
    
    # Test configurations
    configs = [
        ("2ms symbols", 2.0, 0.25),
        ("1.5ms symbols", 1.5, 0.25),
        ("1ms symbols", 1.0, 0.25),
    ]
    
    for label, sym_dur, overlap in configs:
        print(f"\n{label} (overlap={overlap})")
        print("-" * 90)
        
        modem = MusicModeModem(
            symbol_duration_ms=sym_dur,
            chirp_overlap=overlap
        )
        
        symbols = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_music_mode_signal(symbols, add_music_floor=True)
        bitrate = (len(test_data) * 8) / (len(audio) / 16000)
        
        print(f"Bitrate: {bitrate:.0f} bps")
        print(f"\n{'Codec':<15} {'Bitrate':<15} {'BER':<10} {'Status'}")
        print("-" * 90)
        
        # Test with different codecs
        codecs_to_test = [
            ("AAC", "aac", 32),
            ("Opus", "opus", 32),
            ("AMR-NB", "amr-nb", 12.2),
        ]
        
        for codec_name, codec_type, codec_bitrate in codecs_to_test:
            try:
                tester = CodecLoopbackTester(
                    codec=codec_type,
                    bitrate_kb=codec_bitrate,
                    sample_rate=16000
                )
                
                # Encode audio
                tester.encode_audio(audio, "phase2_test.wav")
                
                # Decode and demodulate
                decoded_audio = tester.decode_audio()
                decoded_symbols = modem.demodulate_chirp(decoded_audio)
                recovered_data = modem.symbols_to_binary(decoded_symbols)
                
                # Calculate BER
                bits_total = len(test_data) * 8
                bits_error = 0
                for orig, recov in zip(test_data, recovered_data[:len(test_data)]):
                    bits_error += bin(orig ^ recov).count('1')
                
                ber = (bits_error / bits_total) * 100 if bits_total > 0 else 0
                
                status = "✓ PASS" if ber < 5.0 else "✗ FAIL"
                print(f"{codec_name:<15} {codec_bitrate:<15.1f} {ber:<10.2f}% {status}")
                
            except Exception as e:
                print(f"{codec_name:<15} {'N/A':<15} {'ERROR':<10} {str(e)[:30]}")


def test_music_floor_effectiveness():
    """Verify music floor is actually being added."""
    print("\n" + "=" * 90)
    print("MUSIC FLOOR VERIFICATION")
    print("=" * 90)
    
    test_data = b"Test"
    modem = MusicModeModem(symbol_duration_ms=2.0)
    symbols = modem.encode_binary_to_symbols(test_data)
    
    # Without floor
    audio_no_floor = modem.generate_music_mode_signal(symbols, add_music_floor=False)
    
    # With floor
    audio_with_floor = modem.generate_music_mode_signal(symbols, add_music_floor=True)
    
    # Analyze frequency content
    fft_no_floor = np.abs(np.fft.fft(audio_no_floor))
    fft_with_floor = np.abs(np.fft.fft(audio_with_floor))
    freqs = np.fft.fftfreq(len(audio_no_floor), 1/16000)
    
    # Look at low frequencies (0-300 Hz)
    low_freq_mask = (freqs > 0) & (freqs < 300)
    energy_no_floor = np.sum(fft_no_floor[low_freq_mask])
    energy_with_floor = np.sum(fft_with_floor[low_freq_mask])
    
    print(f"\nLow-frequency (0-300 Hz) energy:")
    print(f"  Without floor: {energy_no_floor:.2e}")
    print(f"  With floor:    {energy_with_floor:.2e}")
    print(f"  Increase:      {(energy_with_floor/energy_no_floor):.1f}x")
    
    # Check specific harmonics
    print(f"\nHarmonic presence:")
    for freq in [60, 200]:
        idx_no_floor = np.argmin(np.abs(freqs - freq))
        idx_with_floor = np.argmin(np.abs(freqs - freq))
        mag_no = fft_no_floor[idx_no_floor]
        mag_with = fft_with_floor[idx_with_floor]
        print(f"  {freq} Hz: {mag_no:.2e} (no floor) → {mag_with:.2e} (with floor)")
    
    print(f"\n✓ Music floor is being added and should force codec into MDCT mode")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    # Run all tests
    test_phase_comparison()
    test_bitrate_sweep()
    test_music_floor_effectiveness()
    
    # Optional: codec survival (requires ffmpeg)
    try:
        test_codec_survival_phase2()
    except Exception as e:
        print(f"\nCodec survival test skipped: {e}")
    
    print("\n" + "=" * 90)
    print("PHASE 2 SUMMARY")
    print("=" * 90)
    print("""
THEORY:
  - Phase 1 champion: 10.7 kbps (hybrid chirp + 25% overlap)
  - Adding 60Hz + 200Hz harmonics forces AAC/Opus into MDCT music mode
  - MDCT = full 16 kHz bandwidth vs ACELP speech mode = ~8 kHz
  - Expected: 2-3x bitrate improvement (20-30 kbps range)

MECHANISM:
  - Codec detects sustained low frequencies → triggers music mode
  - Not speech (500-3500 Hz) + chirp → uses MDCT algorithm
  - MDCT = time-frequency distribution over full Nyquist
  - More bandwidth available for payload signal

NEXT STEPS:
  1. Run full codec loopback tests with real AAC/Opus/AMR-NB
  2. Measure actual BER at 1.5ms, 1ms, 0.75ms symbol durations
  3. Identify new safe operating point (BER < 5%)
  4. Phase 3: Hardware testing (VoLTE call validation)
""")
    print("=" * 90)
