#!/usr/bin/env python3
"""
Debug script: Check modulation/demodulation without codec.

This will help identify if the problem is in modulation/demodulation or the codec.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path

from voice_detector.audio_modem import AudioModem

# Test data
test_data = b"\x12\x34\x56\x78" * 3  # 12 bytes
modem = AudioModem(sample_rate=16000)

# Encode
symbols = modem.encode_binary_to_symbols(test_data)
print(f"Input data ({len(test_data)} bytes): {test_data.hex()}")
print(f"Symbols ({len(symbols)}): {symbols}")

# Modulate
audio = modem.generate_mfsk_signal(symbols, base_freq=200.0, add_voice_noise=False)
print(f"\nModulated audio: {len(audio)} samples, range: [{audio.min():.3f}, {audio.max():.3f}]")

# Save for inspection
out_path = Path("/tmp/debug_modulated.wav")
sf.write(out_path, audio, 16000)
print(f"Saved to: {out_path}")

# Demodulate
symbols_recovered = modem.demodulate_mfsk(audio, base_freq=200.0)
print(f"\nRecovered symbols ({len(symbols_recovered)}): {symbols_recovered}")

# Compare
errors = np.sum(symbols != symbols_recovered)
print(f"Symbol errors: {errors}/{len(symbols)}")

# Convert back to binary
output_data = modem.symbols_to_binary(symbols_recovered)
print(f"Output data: {output_data.hex() if output_data else '(empty)'}")

# Binary comparison
input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
output_bits = np.unpackbits(
    np.frombuffer(output_data, dtype=np.uint8)
) if output_data else np.array([])

max_len = max(len(input_bits), len(output_bits))
input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))

bit_errors = np.sum(input_bits != output_bits)
print(f"Bit errors: {bit_errors}/{max_len} (BER: {bit_errors/max_len:.4f})")
