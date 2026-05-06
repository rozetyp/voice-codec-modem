#!/usr/bin/env python3
"""
Debug script: Deep dive into demodulation energy detection.
"""

from __future__ import annotations

import numpy as np
from voice_detector.audio_modem import AudioModem

# Test data
test_data = b"\x12\x34\x56\x78" * 3
modem = AudioModem(sample_rate=16000)

# Encode
symbols = modem.encode_binary_to_symbols(test_data)
print(f"Input symbols: {symbols[:10]}...")

# Modulate (without voice noise first)
audio = modem.generate_mfsk_signal(symbols, base_freq=200.0, add_voice_noise=False)
print(f"Modulated audio: {len(audio)} samples")

# Manually demodulate and print energies for first few symbols
base_freq = 200.0
freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
samples_per_symbol = modem.samples_per_symbol
sample_rate = modem.sample_rate

print(f"\nDemodulation debug (first 5 symbols):")
print(f"Samples per symbol: {samples_per_symbol}")
print(f"Expected symbols: {symbols[:5]}")
print("\n{:<6} {:<8} {:<8} {:<8} {:<8} {:<8}".format("Sym", "0Hz", "400Hz", "800Hz", "1200Hz", "Best"))

for i in range(min(5, len(audio) // samples_per_symbol)):
    frame = audio[i * samples_per_symbol : (i + 1) * samples_per_symbol]
    
    energies = {}
    for sym, freq in freq_map.items():
        t = np.arange(len(frame)) / sample_rate
        carrier_cos = np.cos(2 * np.pi * freq * t)
        carrier_sin = np.sin(2 * np.pi * freq * t)
        i_comp = np.sum(frame * carrier_cos)
        q_comp = np.sum(frame * carrier_sin)
        energy = i_comp ** 2 + q_comp ** 2
        energies[sym] = energy
    
    best_sym = max(energies, key=energies.get)
    print("{:<6} {:<8.0f} {:<8.0f} {:<8.0f} {:<8.0f} {:<8}".format(
        i, energies[0], energies[1], energies[2], energies[3], int(best_sym)
    ))
