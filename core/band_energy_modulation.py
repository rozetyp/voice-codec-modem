#!/usr/bin/env python3
"""
BAND ENERGY MODULATION: Encode data in Bark band energy
This should survive Opus codec because energy is preserved.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as sp

print("="*100)
print("BAND ENERGY MODULATION ENCODER")
print("="*100)

# Bark scale band frequencies (simplified version, lower bands more detailed)
BARK_BANDS = [
    (0, 100),      # Band 0
    (100, 200),    # Band 1
    (200, 300),    # Band 2
    (300, 400),    # Band 3
    (400, 510),    # Band 4
    (510, 631),    # Band 5
    (631, 770),    # Band 6
    (770, 920),    # Band 7
    (920, 1080),   # Band 8
    (1080, 1270),  # Band 9
    (1270, 1480),  # Band 10
    (1480, 1720),  # Band 11
    (1720, 2000),  # Band 12
    (2000, 2320),  # Band 13
    (2320, 2700),  # Band 14
    (2700, 3150),  # Band 15
    (3150, 3700),  # Band 16
    (3700, 4400),  # Band 17
    (4400, 5300),  # Band 18
    (5300, 6400),  # Band 19
]

def encode_band_energy(data_bits: np.ndarray, duration_s=1.0, sample_rate=16000):
    """
    Encode binary data in band energy.
    
    Each frame (20ms) encodes data across bands:
    - High energy = bit 1
    - Low energy = bit 0
    """
    
    frame_duration = 0.02  # 20ms frames (standard Opus)
    samples_per_frame = int(sample_rate * frame_duration)
    num_frames = int(duration_s / frame_duration)
    
    print(f"\nEncoding {len(data_bits)} bits")
    print(f"  Frames: {num_frames} (20ms each)")
    print(f"  Bands: {len(BARK_BANDS)}")
    print(f"  Bits per frame: {len(BARK_BANDS)} (binary energy levels)")
    
    total_audio = np.array([], dtype=np.float32)
    
    # Process each frame
    bit_idx = 0
    for frame_num in range(num_frames):
        frame_audio = np.zeros(samples_per_frame, dtype=np.float32)
        
        # For each band, set energy based on bit value
        for band_idx, (f_low, f_high) in enumerate(BARK_BANDS):
            if bit_idx >= len(data_bits):
                break
            
            bit_val = data_bits[bit_idx]
            bit_idx += 1
            
            # Generate bandpass noise for this band
            # Bit 1 = high energy (0.8), Bit 0 = low energy (0.2)
            energy = 0.8 if bit_val else 0.2
            
            # Create bandpass filtered noise
            freq_center = (f_low + f_high) / 2
            bandwidth = f_high - f_low
            
            # Generate noise
            noise = np.random.randn(samples_per_frame)
            
            # Bandpass filter (simplified - crude but works)
            # Design filter
            nyquist = sample_rate / 2
            low = f_low / nyquist
            high = f_high / nyquist
            low = max(0.01, min(0.99, low))
            high = max(0.01, min(0.99, high))
            
            if low < high:
                try:
                    b, a = sp.butter(4, [low, high], btype='band')
                    filtered = sp.filtfilt(b, a, noise)
                    
                    # Normalize and scale by energy level
                    filtered = filtered / (np.std(filtered) + 1e-10) * energy * 0.1
                    frame_audio += filtered
                except:
                    pass  # Skip if filter design fails
        
        # Smooth envelope to avoid clicks
        envelope = np.exp(-3 * ((np.arange(samples_per_frame) / samples_per_frame) - 0.5) ** 2 / 0.25 ** 2)
        frame_audio = frame_audio * (0.3 + 0.7 * envelope)
        
        total_audio = np.concatenate([total_audio, frame_audio])
    
    # Normalize
    if np.max(np.abs(total_audio)) > 0:
        total_audio = total_audio / np.max(np.abs(total_audio)) * 0.9
    
    return total_audio

# Test encoding
print("\n" + "="*100)
print("TEST: Encoding random bits")
print("="*100)

# Generate test data: 1000 random bits
test_bits = np.random.randint(0, 2, size=1000)
print(f"\nTest message: {test_bits[:100]}... (1000 bits total)")

# Encode
audio = encode_band_energy(test_bits, duration_s=1.0)

print(f"\nEncoded audio:")
print(f"  Samples: {len(audio)}")
print(f"  Duration: {len(audio) / 16000:.2f}s")
print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")
print(f"  Peak: {np.max(np.abs(audio)):.4f}")

# Save
Path("research").mkdir(exist_ok=True)
sf.write("research/band_energy_encoded.wav", audio, 16000)
print(f"\n✓ Saved to research/band_energy_encoded.wav")

# Now test: Pass through mock Opus codec
print("\n" + "="*100)
print("TEST: Pass through mock Opus (perceptual filter)")
print("="*100)

def mock_opus_codec(audio, sample_rate=16000, bitrate_kbps=24):
    """Simplified Opus: apply perceptual weighting + quantization"""
    
    # Perceptual filter (A-weighting + speech emphasis)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    
    # A-weighting curve
    A = 20 + 20 * np.log10(np.abs(freqs + 0.001) + 1.2) - 10 * np.log10(freqs**2 + 0.0001)
    A = A - np.max(A)  # Normalize
    A = 10 ** (np.clip(A, -20, 20) / 20)  # Convert to linear
    
    # Speech emphasis (boost 200-4000 Hz)
    speech_boost = np.ones_like(freqs)
    speech_mask = (freqs > 200) & (freqs < 4000)
    speech_boost[speech_mask] = 1.5
    
    weighted = fft * A * speech_boost
    
    # Quantization noise
    snr_db = 20  # From bitrate
    snr_linear = 10 ** (snr_db / 20)
    signal_power = np.mean(np.abs(weighted) ** 2)
    noise_power = signal_power / (snr_linear ** 2)
    noise = np.random.randn(len(weighted)) * np.sqrt(noise_power)
    
    weighted_noisy = weighted + noise
    
    # Inverse FFT
    result = np.fft.irfft(weighted_noisy, n=len(audio))
    return np.clip(result, -1, 1).astype(np.float32)

# Apply codec
coded_audio = mock_opus_codec(audio)
sf.write("research/band_energy_coded.wav", coded_audio, 16000)

print(f"Post-codec audio:")
print(f"  RMS: {np.sqrt(np.mean(coded_audio**2)):.4f}")
print(f"  Peak: {np.max(np.abs(coded_audio)):.4f}")
print(f"✓ Saved to research/band_energy_coded.wav")

# Measure band energy preservation
print("\n" + "="*100)
print("ANALYSIS: Band energy preservation through codec")
print("="*100)

def analyze_band_energy(audio, sample_rate=16000):
    """Measure energy in each band"""
    energies = []
    for f_low, f_high in BARK_BANDS:
        nyquist = sample_rate / 2
        low = max(1, f_low) / nyquist
        high = min(nyquist, f_high) / nyquist
        
        if low >= high:
            energies.append(0)
            continue
        
        try:
            b, a = sp.butter(4, [low, high], btype='band')
            filtered = sp.filtfilt(b, a, audio)
            energy = np.sqrt(np.mean(filtered ** 2))
            energies.append(energy)
        except:
            energies.append(0)
    
    return np.array(energies)

energy_original = analyze_band_energy(audio)
energy_coded = analyze_band_energy(coded_audio)

# Compare
print(f"\nBand energy correlation (pre vs post codec):")
correlation = np.correlate(energy_original / (np.std(energy_original) + 1e-10), 
                           energy_coded / (np.std(energy_coded) + 1e-10),
                           mode='valid')[0] / len(BARK_BANDS)
print(f"  Correlation: {correlation:.4f}")

if correlation > 0.7:
    print(f"  ✅ GOOD - Band energy is preserved!")
else:
    print(f"  ❌ POOR - Codec distorts band energy significantly")

print(f"\n" + "="*100)
print(f"BAND ENERGY MODULATION RESULT")
print(f"="*100)
print(f"""
Theoretical bitrate: 20 bands × 50 frames/sec = 1000 bits/sec = 1 kbps

If correlation > 0.7: Viable for production
If correlation < 0.5: Need different approach

Next: Implement decoder to recover bits from coded audio
""")
