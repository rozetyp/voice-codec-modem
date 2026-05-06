#!/usr/bin/env python3
"""
Generate audio samples of each modulation technique.
Listen to the evolution from discrete tones to music-like chirps.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path

from voice_detector.audio_modem import AudioModem
from voice_detector.radical_modem import ChirpModem

# Create output directory
output_dir = Path("audio_samples")
output_dir.mkdir(exist_ok=True)

test_data = b"VOCAL MODEM 2026"  # Will encode this

print("\n" + "=" * 80)
print("GENERATING AUDIO SAMPLES - LISTEN TO THE MODEM")
print("=" * 80)

# 1. Traditional MFSK (4-ary, discrete tones)
print("\n1. TRADITIONAL MFSK (Discrete Tones)")
print("   └─ Sounds like: beeping tones (like old phone modems)")
modem_mfsk = AudioModem(symbol_duration_ms=20.0)
symbols_mfsk = modem_mfsk.encode_binary_to_symbols(test_data)
audio_mfsk = modem_mfsk.generate_mfsk_signal(symbols_mfsk, base_freq=200.0, add_voice_noise=False)
sf.write(output_dir / "1_traditional_mfsk_20ms.wav", audio_mfsk, 16000)
print(f"   ✓ Saved: 1_traditional_mfsk_20ms.wav ({len(audio_mfsk)/16000:.2f}s)")

# 2. MFSK with voice noise
print("\n2. TRADITIONAL MFSK + VOICE CHARACTERISTICS")
print("   └─ Sounds like: beeping tones with slight pitch variation")
audio_mfsk_voice = modem_mfsk.generate_mfsk_signal(symbols_mfsk, base_freq=200.0, add_voice_noise=True)
sf.write(output_dir / "2_mfsk_with_voice_noise_20ms.wav", audio_mfsk_voice, 16000)
print(f"   ✓ Saved: 2_mfsk_with_voice_noise_20ms.wav ({len(audio_mfsk_voice)/16000:.2f}s)")

# 3. Chirp (frequency sweeps)
print("\n3. CHIRP MODULATION (Frequency Sweeps)")
print("   └─ Sounds like: sliding whistles (like Star Trek or whale song)")
modem_chirp = ChirpModem(symbol_duration_ms=20.0)
symbols_chirp = modem_chirp.encode_binary_to_symbols(test_data)
audio_chirp = modem_chirp.generate_chirp_signal(symbols_chirp)
sf.write(output_dir / "3_chirp_20ms.wav", audio_chirp, 16000)
print(f"   ✓ Saved: 3_chirp_20ms.wav ({len(audio_chirp)/16000:.2f}s)")

# 4. Faster Chirp (5ms symbols)
print("\n4. FAST CHIRP (5ms Symbols) - 3.2 kbps")
print("   └─ Sounds like: rapid sliding whistles, metallic")
modem_chirp_fast = ChirpModem(symbol_duration_ms=5.0)
symbols_chirp_fast = modem_chirp_fast.encode_binary_to_symbols(test_data)
audio_chirp_fast = modem_chirp_fast.generate_chirp_signal(symbols_chirp_fast)
sf.write(output_dir / "4_chirp_5ms_fast.wav", audio_chirp_fast, 16000)
print(f"   ✓ Saved: 4_chirp_5ms_fast.wav ({len(audio_chirp_fast)/16000:.2f}s)")

# 5. Ultra-fast Chirp (2ms symbols) - 8 kbps
print("\n5. ULTRA-FAST CHIRP (2ms Symbols) - 8 kbps")
print("   └─ Sounds like: very rapid chirps, almost like static with pattern")
modem_chirp_ultra = ChirpModem(symbol_duration_ms=2.0)
symbols_chirp_ultra = modem_chirp_ultra.encode_binary_to_symbols(test_data)
audio_chirp_ultra = modem_chirp_ultra.generate_chirp_signal(symbols_chirp_ultra)
sf.write(output_dir / "5_chirp_2ms_ultra.wav", audio_chirp_ultra, 16000)
print(f"   ✓ Saved: 5_chirp_2ms_ultra.wav ({len(audio_chirp_ultra)/16000:.2f}s)")

# 6. Chirp with background "music" (forces music mode in codec)
print("\n6. CHIRP + MUSIC LAYER (Codec Mode Forcing)")
print("   └─ Sounds like: chirps with low bass hum underneath")
t = np.arange(len(audio_chirp_ultra)) / 16000
music_layer = 0.08 * np.sin(2 * np.pi * 60 * t) + 0.08 * np.sin(2 * np.pi * 200 * t)
audio_hybrid_proto = audio_chirp_ultra + music_layer
max_val = np.max(np.abs(audio_hybrid_proto))
if max_val > 0:
    audio_hybrid_proto = audio_hybrid_proto / max_val * 0.9
sf.write(output_dir / "6_chirp_music_mode_forcing.wav", audio_hybrid_proto.astype(np.float32), 16000)
print(f"   ✓ Saved: 6_chirp_music_mode_forcing.wav ({len(audio_hybrid_proto)/16000:.2f}s)")

# 7. Just the "music floor" (what we add to codec)
print("\n7. MUSIC FLOOR (Background Layer Only)")
print("   └─ Sounds like: low frequency hum (60Hz + 200Hz)")
music_only = 0.15 * np.sin(2 * np.pi * 60 * t) + 0.15 * np.sin(2 * np.pi * 200 * t)
sf.write(output_dir / "7_music_floor_only.wav", music_only.astype(np.float32), 16000)
print(f"   ✓ Saved: 7_music_floor_only.wav ({len(music_only)/16000:.2f}s)")

# 8. Progression: MFSK -> Chirp (comparison)
print("\n8. SIDE-BY-SIDE: MFSK then CHIRP (for comparison)")
print("   └─ Listen to the difference: discrete vs smooth")
comparison = np.concatenate([audio_mfsk, np.zeros(8000), audio_chirp_ultra])
sf.write(output_dir / "8_comparison_mfsk_vs_chirp.wav", comparison.astype(np.float32), 16000)
print(f"   ✓ Saved: 8_comparison_mfsk_vs_chirp.wav ({len(comparison)/16000:.2f}s)")

print("\n" + "=" * 80)
print("AUDIO SUMMARY")
print("=" * 80)
print(f"""
All samples saved to: {output_dir}/

PROGRESSION:
1. Traditional MFSK ........... Old modem sound (1980s-90s)
2. MFSK + Voice ............. Slightly more organic
3. Chirp 20ms ............... Swept tones (much smoother)
4. Chirp 5ms ................ Faster sweeps (higher bitrate)
5. Chirp 2ms ................ Ultra-fast (8 kbps champion) ⭐
6. Chirp + Music Mode ....... Hybrid (what Phase 2 uses)
7. Music Floor .............. Just the background harmonics
8. Comparison ............... MFSK vs Chirp back-to-back

WHAT TO LISTEN FOR:
- Samples 1-2: Discrete tones, clearly separated frequencies
- Samples 3-5: Smooth frequency sweeps (chirps), increasingly rapid
- Sample 6: Chirp with low bass hum (the "music floor")
- Sample 8: Direct comparison shows why chirp wins (less metallic)

CODEC BEHAVIOR:
- MFSK sounds harsh because discrete tones trigger speech codec filtering
- Chirps sound smoother because sweeps trigger music codec mode
- This is WHY chirps achieve 4x better bitrate!
""")

print("\n💡 To listen:")
print(f"   open audio_samples/")
print(f"   or: afplay audio_samples/5_chirp_2ms_ultra.wav")
print("\n" + "=" * 80)
