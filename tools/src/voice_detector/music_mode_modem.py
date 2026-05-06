"""
Phase 2: Music Mode Forcing - Hybrid Chirp with Harmonic Layer

Forces audio codecs into MDCT (music mode) instead of ACELP (speech mode)
by adding 50-100 Hz harmonics underneath the chirp payload.

Music mode = full 16 kHz bandwidth instead of ~4-8 kHz speech bandwidth
Expected: 2-3x bitrate improvement over Phase 1 champion (10.7 kbps)

Architecture:
  Binary Data
      ↓
  Encode symbols (4-ary)
      ↓
  Generate hybrid chirp signal (payload)
      ↓
  ADD MUSIC FLOOR (50Hz + 200Hz harmonics) ← NEW
      ↓
  Audio (forces codec into music mode)
      ↓
  [Codec encodes with MDCT not ACELP]
      ↓
  Demodulate chirp + ignore harmonics
      ↓
  Recover binary data
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from pathlib import Path


class MusicModeModem:
    """Hybrid chirp modem with music floor forcing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        symbol_duration_ms: float = 2.0,
        chirp_overlap: float = 0.25,
        music_floor_frequencies: list[float] | None = None,
        music_floor_amplitude: float = 0.08,
    ):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            symbol_duration_ms: Duration per symbol (milliseconds)
            chirp_overlap: Overlap ratio for hybrid chirp (0.0 = no overlap)
            music_floor_frequencies: Harmonics to add [60, 200] Hz typical
            music_floor_amplitude: Amplitude of music floor (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.chirp_overlap = chirp_overlap
        self.music_floor_frequencies = music_floor_frequencies or [60, 200]
        self.music_floor_amplitude = music_floor_amplitude
        
        self.samples_per_symbol = int(sample_rate * symbol_duration_ms / 1000.0)
        
        # Chirp band (as in Phase 1)
        self.base_freq = 200.0  # Hz
        self.freq_spacing = 400.0  # Hz
        self.num_symbols = 4  # 4-ary (2 bits per symbol)
        
    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert binary data to 4-ary symbols."""
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)
    
    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """Convert 4-ary symbols back to binary data."""
        result = bytearray()
        for i in range(0, len(symbols), 4):
            if i + 3 < len(symbols):
                byte_val = (
                    (symbols[i] << 6) |
                    (symbols[i+1] << 4) |
                    (symbols[i+2] << 2) |
                    symbols[i+3]
                )
                result.append(byte_val)
        return bytes(result)
    
    def add_music_floor(self, audio: np.ndarray) -> np.ndarray:
        """
        Add low-frequency music harmonics to force codec into music mode.
        
        Codec detection:
        - Detects sustained low frequencies (50-200 Hz) → MDCT music mode
        - Only speech frequencies (500-3500 Hz) → ACELP speech mode
        
        By adding 60 Hz + 200 Hz, we trigger music mode detection.
        Music mode = MDCT algorithm = full bandwidth (16 kHz) instead of ~8 kHz
        """
        t = np.arange(len(audio)) / self.sample_rate
        music = np.zeros_like(audio)
        
        for freq in self.music_floor_frequencies:
            music += self.music_floor_amplitude * np.sin(2 * np.pi * freq * t)
        
        # Blend and normalize to avoid clipping
        combined = audio + music
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.95
        
        return combined
    
    def generate_chirp_signal(self, symbols: np.ndarray) -> np.ndarray:
        """
        Generate chirp signal (frequency sweeps) per symbol.
        Linear frequency sweep from base_freq to base_freq + freq_spacing.
        """
        num_symbols = len(symbols)
        
        # Calculate overlap-based timing
        step_samples = int(self.samples_per_symbol * (1.0 - self.chirp_overlap))
        total_samples = self.samples_per_symbol + (num_symbols - 1) * step_samples
        
        audio = np.zeros(total_samples)
        window = signal.windows.hann(self.samples_per_symbol)
        
        for i, sym in enumerate(symbols):
            # Chirp frequency: base_freq + sym * freq_spacing
            f0 = self.base_freq + sym * self.freq_spacing
            f1 = f0 + self.freq_spacing  # Sweep to next frequency band
            
            # Linear chirp: phase(t) = 2π(f0*t + (f1-f0)*t²/(2*duration))
            duration = self.samples_per_symbol / self.sample_rate
            t = np.arange(self.samples_per_symbol) / self.sample_rate
            phase = 2 * np.pi * (
                f0 * t + (f1 - f0) * t**2 / (2 * duration)
            )
            
            chirp_symbol = np.cos(phase) * window
            
            # Place with overlap
            start_idx = i * step_samples
            end_idx = start_idx + self.samples_per_symbol
            audio[start_idx:end_idx] += chirp_symbol
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def generate_music_mode_signal(
        self,
        symbols: np.ndarray,
        add_music_floor: bool = True,
    ) -> np.ndarray:
        """
        Generate complete signal with optional music floor.
        
        Args:
            symbols: 4-ary symbols to modulate
            add_music_floor: If True, add harmonics to force music mode
        
        Returns:
            Audio signal ready for codec
        """
        # Generate chirp payload
        chirp = self.generate_chirp_signal(symbols)
        
        # Add music floor if requested
        if add_music_floor:
            return self.add_music_floor(chirp)
        else:
            return chirp
    
    def demodulate_chirp(self, audio: np.ndarray) -> np.ndarray:
        """
        Demodulate chirp signal back to symbols using matched filtering.
        Each symbol is a linear frequency sweep from f0 to f1.
        """
        symbols = []
        step_samples = int(self.samples_per_symbol * (1.0 - self.chirp_overlap))
        num_possible_symbols = int(
            (len(audio) - self.samples_per_symbol) / step_samples + 1
        )
        
        # Pre-compute chirp templates for each symbol
        duration = self.samples_per_symbol / self.sample_rate
        t = np.arange(self.samples_per_symbol) / self.sample_rate
        window = signal.windows.hann(self.samples_per_symbol)
        
        templates = []
        for sym in range(self.num_symbols):
            f0 = self.base_freq + sym * self.freq_spacing
            f1 = f0 + self.freq_spacing
            
            phase = 2 * np.pi * (
                f0 * t + (f1 - f0) * t**2 / (2 * duration)
            )
            chirp_template = np.cos(phase) * window
            templates.append(chirp_template)
        
        # Sliding window demodulation using matched filters
        for i in range(num_possible_symbols):
            start_idx = i * step_samples
            end_idx = start_idx + self.samples_per_symbol
            
            if end_idx > len(audio):
                break
            
            segment = audio[start_idx:end_idx]
            
            # Correlate with all templates
            best_sym = 0
            best_correlation = 0
            
            for sym, template in enumerate(templates):
                # Normalized cross-correlation
                correlation = np.correlate(segment, template, mode='valid')
                if len(correlation) > 0:
                    corr_value = np.max(np.abs(correlation))
                    if corr_value > best_correlation:
                        best_correlation = corr_value
                        best_sym = sym
            
            symbols.append(best_sym)
        
        return np.array(symbols, dtype=np.uint8)
    
    def add_awgn(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Additive White Gaussian Noise at specified SNR."""
        signal_power = np.mean(audio**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise


if __name__ == "__main__":
    # Quick test
    modem = MusicModeModem(symbol_duration_ms=2.0, chirp_overlap=0.25)
    test_data = b"PHASE 2"
    
    symbols = modem.encode_binary_to_symbols(test_data)
    print(f"Symbols: {symbols}")
    
    # Without music floor
    audio_no_floor = modem.generate_music_mode_signal(symbols, add_music_floor=False)
    print(f"Audio (no floor): {len(audio_no_floor)} samples, {len(audio_no_floor)/16000:.3f}s")
    
    # With music floor
    audio_with_floor = modem.generate_music_mode_signal(symbols, add_music_floor=True)
    print(f"Audio (with floor): {len(audio_with_floor)} samples, {len(audio_with_floor)/16000:.3f}s")
    
    # Demodulate
    decoded_symbols = modem.demodulate_chirp(audio_with_floor)
    recovered_data = modem.symbols_to_binary(decoded_symbols)
    print(f"Recovered: {recovered_data}")
    print(f"Match: {recovered_data == test_data}")
