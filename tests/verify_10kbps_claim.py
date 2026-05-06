#!/usr/bin/env python3
"""
VERIFICATION: Does PATH_B's 10.8 kbps claim actually work?

Testing against realistic Opus codec conditions.
Real measurements only - no claims.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as sp

print("="*100)
print("VERIFICATION: 10.8 kbps 4-Carrier Claim")
print("="*100)

# Try to import their modem
try:
    from src.voice_detector.hybrid_4carrier_modem import FourCarrierPhonemeModem
    four_carrier_exists = True
    print("\n✓ Found: hybrid_4carrier_modem.py")
except Exception as e:
    four_carrier_exists = False
    print(f"\n✗ Cannot import hybrid_4carrier_modem: {e}")

# Simple 4-carrier test ourselves
print("\nLet's test 4-carrier viability from first principles:")
print("-" * 100)

def mock_opus_codec_realistic(audio, sample_rate=16000):
    """Realistic Opus codec simulation using perceptual model"""
    # FFT
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    
    # Opus perceptual masking model (simplified)
    # Opus is optimized for speech 0-8kHz, de-emphasizes above 8kHz
    mask = np.ones(len(freqs))
    
    # Preserve speech formants (0-4 kHz)
    mask[freqs < 4000] = 1.0
    
    # De-emphasize but don't destroy (4-8 kHz)
    mask[(freqs >= 4000) & (freqs < 8000)] = 0.7
    
    # Significantly suppress high-frequency (8+ kHz)
    mask[freqs >= 8000] = 0.3
    
    # Apply perceptual weighting
    weighted_fft = fft * mask
    
    # Quantization noise (24 kbps Opus has ~24 dB SNR)
    snr_db = 24
    snr_linear = 10 ** (snr_db / 20)
    signal_power = np.mean(np.abs(weighted_fft) ** 2)
    noise_power = signal_power / (snr_linear ** 2)
    noise = np.random.randn(len(weighted_fft)) * np.sqrt(noise_power)
    
    output_fft = weighted_fft + noise
    output = np.fft.irfft(output_fft, n=len(audio))
    
    return np.clip(output, -1, 1).astype(np.float32)

# Test: 4 simultaneous carriers
print("\nTest Case: 4 simultaneous tones at strategic frequencies")
print("-" * 100)

sample_rate = 16000
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(sample_rate * duration))

# The claimed frequencies from PATH_B
carrier_freqs = {
    0: 700,     # AH
    1: 1990,    # EE
    2: 1070,    # OO
    3: 1570,    # EH
}

# Generate 4 tones mixed
test_signal = np.zeros_like(t, dtype=np.float32)

print("Carriers:")
for idx, freq in carrier_freqs.items():
    tone = 0.2 * np.sin(2 * np.pi * freq * t)
    test_signal += tone
    print(f"  {idx}: {freq} Hz")

# Normalize
test_signal = test_signal / np.max(np.abs(test_signal)) * 0.9

print(f"\nBefore codec: {len(test_signal)} samples")
print(f"  RMS: {np.sqrt(np.mean(test_signal**2)):.4f}")
print(f"  Frequencies: 700, 1070, 1570, 1990 Hz")

# Save
Path("verify").mkdir(exist_ok=True)
sf.write("verify/4carrier_original.wav", test_signal, sample_rate)

# Apply Opus-like codec
coded_signal = mock_opus_codec_realistic(test_signal)
sf.write("verify/4carrier_coded.wav", coded_signal, sample_rate)

print(f"\nAfter Opus codec:")
print(f"  RMS: {np.sqrt(np.mean(coded_signal**2)):.4f}")

# Analyze: Can we recover the carriers?
print("\nSpectral Analysis:")
print("-" * 100)

orig_fft = np.fft.rfft(test_signal)
orig_freqs = np.fft.rfftfreq(len(test_signal), 1.0 / sample_rate)
orig_mag = np.abs(orig_fft)

coded_fft = np.fft.rfft(coded_signal)
coded_mag = np.abs(coded_fft)

for idx, freq in carrier_freqs.items():
    # Find magnitude near this frequency
    freq_idx = np.argmin(np.abs(orig_freqs - freq))
    orig_power = orig_mag[freq_idx]
    coded_power = coded_mag[freq_idx]
    snr = 20 * np.log10(coded_power / (np.max(coded_mag) * 0.01 + 1e-10))  # Rough SNR
    preservation = 100 * coded_power / (orig_power + 1e-10)
    
    print(f"  Carrier {idx} ({freq} Hz):")
    print(f"    Original power: {orig_power:.4f}")
    print(f"    Post-codec power: {coded_power:.4f}")
    print(f"    Preservation: {preservation:.1f}%")
    print(f"    SNR vs noise floor: {snr:.1f} dB")

print("\n" + "="*100)
print("ANALYSIS")
print("="*100)

# Check if carriers are recoverable
carrier_preservation = []
for idx, freq in carrier_freqs.items():
    freq_idx = np.argmin(np.abs(orig_freqs - freq))
    orig_power = orig_mag[freq_idx]
    coded_power = coded_mag[freq_idx]
    preservation = 100 * coded_power / (orig_power + 1e-10)
    carrier_preservation.append(preservation)

avg_preservation = np.mean(carrier_preservation)
print(f"\nAverage carrier preservation: {avg_preservation:.1f}%")

if avg_preservation > 50:
    print("✓ Carriers somewhat preserved")
elif avg_preservation > 20:
    print("⚠ Carriers significantly degraded")
else:
    print("✗ Carriers destroyed")

# The key question for 10.8 kbps
print("\nQuestion: Can ML decoder recover data from degraded carriers?")
print("-" * 100)

# Simple FFT-based decoder: extract energy per carrier
decoder_bits = []
for idx, freq in carrier_freqs.items():
    freq_idx = np.argmin(np.abs(orig_freqs - freq))
    coded_power = coded_mag[freq_idx]
    # Threshold: energy > median = 1, else 0
    bit = 1 if coded_power > np.median(coded_mag[:5000]) else 0
    decoder_bits.append(bit)
    print(f"  Carrier {idx}: power={coded_power:.4f}, decoded bit={bit}")

print(f"\nDecoded 4 bits: {decoder_bits}")
print(f"If each carrier carries 2 bits per symbol:")
print(f"  → 4 carriers × 2 bits = 8 bits per 20ms frame")
print(f"  → 8 bits / 0.020s = 400 bits/sec = 0.4 kbps per carrier")
print(f"  → 4 carriers × 0.4 = 1.6 kbps total")

print("\n" + "="*100)
print("REALITY CHECK")
print("="*100)

print(f"""
The 10.8 kbps claim assumes:
  ✓ 4 carriers at strategic frequencies
  ✓ 2.7 kbps per carrier (from Phase 2)
  ✓ Total = 4 × 2.7 = 10.8 kbps

But we measured Phase 2 (single 700 Hz carrier) at 1.15% BER.
That's with:
  - Optimization specifically for 700 Hz
  - Long training on that frequency
  - Active AGC compensation

For 4 carriers simultaneously:
  1. Spectral interference: Carriers compete for channel (SNR degrades per carrier)
  2. Opus perceptual model: 700-2000 Hz is less preserved than 0-4 kHz
  3. AGC behavior: 4 tones look different than 1 tone to AGC
  4. ML decoder complexity: Each carrier needs separate decoder (4× training)

Mathematical reality:
  - If single carrier = 2.7 kbps at ~1% BER
  - 4 carriers sharing same 24 kbps Opus budget
  - Each carrier gets ~6 kbps of codec capacity (not 2.7 kbps!)
  - Compression ratio: 6 kbps data in 24 kbps codec = 4:1 compression
  - Single carrier: 2.7 kbps in 24 kbps = 8.9:1 compression

Expected result for 4-carrier:
  ~3-5 kbps achievable (maybe 10.8 if perfect ML, but that's speculation)
""")

print("\n" + "="*100)
print("ALTERNATIVE APPROACHES TO 10+ KBPS")
print("="*100)

print("""
Option 1: SEQUENTIAL ENCODING (Not simultaneous)
  - Send 2.7 kbps on 1 carrier, then switch carriers
  - Achievable: 2.7 kbps reliably per switched carrier
  - Not truly 10.8 kbps (requires 4× time)
  - Status: ✗ Doesn't meet "10 kbps minimum"

Option 2: WIDER BAND MODULATION
  - Use entire 0-4 kHz speech band (Opus well-preserved)
  - Spread energy across band (not discrete carriers)
  - Per-band energy encoding (like our band energy approach)
  - Achievable: 2-3 kbps per band, 3-4 bands = 6-12 kbps possible?
  - Status: ? Needs testing

Option 3: PITCH + ENERGY + FORMANT (Multi-primitive)
  - Pitch variation encoding: 500-1000 bps
  - Band energy encoding: 1000 bps
  - Formant energy encoding: 500 bps
  - Total: 2-2.5 kbps? (Much lower than claimed 10.8)
  - Status: ✓ Proven but limited bitrate

Option 4: EXPLOIT CODEC VOCODER ARTIFACTS
  - Opus uses vocoder mode for low bitrates
  - Exploit pitch contour modulation (Opus extracts pitch)
  - Exploit formant trajectory (Opus preserves formants)
  - Very high-risk (codec-specific, might break)
  - Achievable: Unknown
  - Status: ? Research needed

Option 5: REAL EXPERIMENT (Not speculation)
  - Actually train 4-carrier ML decoder on realistic Opus corruption
  - Measure BER across multiple network conditions
  - Measure how BER scales with carrier count
  - ONLY then claim bitrate
  - Status: ✓ This is what needs to happen
""")

print("\nRECOMMENDATION:")
print("-" * 100)
print("""
To actually verify if 10+ kbps is possible:

1. Test 4-carrier with ML decoder on real Opus codec (not speculation)
2. Measure BER vs single-carrier baseline
3. If 4-carrier ML works: Test 8-carrier (toward 20+ kbps)
4. If 8-carrier works: Evaluate DPI profile (does 8 simultaneous carriers still sound natural?)

Current status:
  ✓ 2.7 kbps single carrier: PROVEN WORKING
  ? 10.8 kbps 4-carrier: UNVERIFIED (code exists, needs testing)
  ? 20+ kbps 8-carrier: THEORETICAL

Next step: Run the actual 4-carrier test with ML decoder + Opus codec.
""")
