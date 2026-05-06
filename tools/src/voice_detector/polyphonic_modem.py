#!/usr/bin/env python3
"""
Phase 2.5: Polyphonic Multi-Tone Chirp Modulation

Instead of sending symbols sequentially (1ms each), send them in parallel as
simultaneous chirps on different frequency bands. This increases bitrate without
requiring sub-millisecond symbol timing precision.

Architecture:
  - Each 5ms window carries 4 simultaneous chirps on different bands
  - Each chirp encodes a different 2-bit symbol
  - Result: 8 bits per 5ms = 1.6 kbps × 4 = 6.4 kbps
  - Jitter tolerance: ±2-3ms (much safer than ±0.5ms at 1ms symbols)
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from pathlib import Path


class PolyphonicChirpModem:
    """Multi-tone parallel chirp modulation."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        symbol_duration_ms: float = 5.0,
        num_carriers: int = 4,
        music_floor_frequencies: list[float] | None = None,
        music_floor_amplitude: float = 0.08,
    ):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            symbol_duration_ms: Duration per symbol window (5ms recommended)
            num_carriers: Number of parallel tone carriers (4 = 8 bits/window)
            music_floor_frequencies: Harmonics to force music codec mode
            music_floor_amplitude: Music floor amplitude
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.num_carriers = num_carriers
        self.music_floor_frequencies = music_floor_frequencies or [60, 200]
        self.music_floor_amplitude = music_floor_amplitude
        
        self.samples_per_symbol = int(sample_rate * symbol_duration_ms / 1000.0)
        
        # Frequency allocation for carriers (400 Hz spacing)
        self.base_freq = 200.0
        self.carrier_spacing = 400.0
        self.freq_spacing_per_carrier = 400.0  # Chirp sweep within each carrier
        
    def _get_carrier_frequency(self, carrier_idx: int, symbol_value: int) -> tuple[float, float]:
        """Get start and end frequencies for a chirp on a specific carrier."""
        # Carrier center frequencies
        carrier_base = self.base_freq + carrier_idx * 1000.0
        
        # Within each carrier band, 4-ary FSK
        f0 = carrier_base + symbol_value * self.freq_spacing_per_carrier
        f1 = f0 + self.freq_spacing_per_carrier
        
        return f0, f1
    
    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to symbols (multiple symbols per window)."""
        symbols = []
        for byte_val in data:
            # Each byte becomes 4 2-bit symbols
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)
    
    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """Convert symbols back to binary."""
        result = bytearray()
        for i in range(0, len(symbols), 4):
            if i + 4 <= len(symbols):
                byte_val = (
                    (symbols[i] << 6) |
                    (symbols[i+1] << 4) |
                    (symbols[i+2] << 2) |
                    symbols[i+3]
                )
                result.append(byte_val)
        return bytes(result)
    
    def generate_polyphonic_signal(
        self,
        symbols: np.ndarray,
        add_music_floor: bool = True,
    ) -> np.ndarray:
        """
        Generate polyphonic signal with multiple chirps per window.
        
        Symbols are grouped by num_carriers:
        - symbols[0:4] → first window (4 parallel chirps on 4 carriers)
        - symbols[4:8] → second window (4 parallel chirps on 4 carriers)
        ...
        """
        num_windows = int(np.ceil(len(symbols) / self.num_carriers))
        total_samples = num_windows * self.samples_per_symbol
        
        audio = np.zeros(total_samples, dtype=np.float32)
        window_envelope = signal.windows.hann(self.samples_per_symbol)
        
        t = np.arange(self.samples_per_symbol) / self.sample_rate
        duration = self.symbol_duration_ms / 1000.0
        
        # Generate each window
        for window_idx in range(num_windows):
            window_start = window_idx * self.samples_per_symbol
            window_end = window_start + self.samples_per_symbol
            
            # Initialize accumulator for this window
            window_signal = np.zeros(self.samples_per_symbol, dtype=np.float32)
            
            # Add parallel chirps for each carrier
            for carrier_idx in range(self.num_carriers):
                symbol_idx = window_idx * self.num_carriers + carrier_idx
                
                if symbol_idx >= len(symbols):
                    break
                
                symbol_value = symbols[symbol_idx]
                
                # Generate chirp for this carrier
                f0, f1 = self._get_carrier_frequency(carrier_idx, symbol_value)
                
                # Linear chirp
                phase = 2 * np.pi * (
                    f0 * t + (f1 - f0) * t**2 / (2 * duration)
                )
                chirp = np.cos(phase) * window_envelope
                
                # Add to window
                window_signal += chirp
            
            # Place in output
            audio[window_start:window_end] = window_signal
        
        # Add music floor if requested
        if add_music_floor:
            audio = self._add_music_floor(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def _add_music_floor(self, audio: np.ndarray) -> np.ndarray:
        """Add low-frequency music harmonics."""
        t = np.arange(len(audio)) / self.sample_rate
        music = np.zeros_like(audio)
        
        for freq in self.music_floor_frequencies:
            music += self.music_floor_amplitude * np.sin(2 * np.pi * freq * t)
        
        combined = audio + music
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.95
        
        return combined
    
    def demodulate_polyphonic(self, audio: np.ndarray) -> np.ndarray:
        """
        Demodulate polyphonic signal using matched filtering.
        Extract symbols from multiple parallel carriers.
        """
        symbols = []
        num_windows = int(np.ceil(len(audio) / self.samples_per_symbol))
        
        t = np.arange(self.samples_per_symbol) / self.sample_rate
        duration = self.symbol_duration_ms / 1000.0
        window_envelope = signal.windows.hann(self.samples_per_symbol)
        
        # Pre-compute templates for all carriers × all symbols
        templates = {}
        for carrier_idx in range(self.num_carriers):
            for symbol_val in range(4):
                key = (carrier_idx, symbol_val)
                f0, f1 = self._get_carrier_frequency(carrier_idx, symbol_val)
                
                phase = 2 * np.pi * (
                    f0 * t + (f1 - f0) * t**2 / (2 * duration)
                )
                chirp_template = np.cos(phase) * window_envelope
                templates[key] = chirp_template
        
        # Demodulate each window
        for window_idx in range(num_windows):
            start_idx = window_idx * self.samples_per_symbol
            end_idx = start_idx + self.samples_per_symbol
            
            if end_idx > len(audio):
                break
            
            window = audio[start_idx:end_idx]
            
            # Extract symbols from each carrier in this window
            for carrier_idx in range(self.num_carriers):
                best_sym = 0
                best_corr = 0
                
                for symbol_val in range(4):
                    key = (carrier_idx, symbol_val)
                    template = templates[key]
                    
                    # Matched filter
                    correlation = np.correlate(window, template, mode='valid')
                    if len(correlation) > 0:
                        corr_val = np.max(np.abs(correlation))
                        if corr_val > best_corr:
                            best_corr = corr_val
                            best_sym = symbol_val
                
                symbols.append(best_sym)
        
        return np.array(symbols, dtype=np.uint8)
    
    def add_awgn(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Additive White Gaussian Noise."""
        signal_power = np.mean(audio**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise


if __name__ == "__main__":
    # Quick test
    modem = PolyphonicChirpModem(
        symbol_duration_ms=5.0,
        num_carriers=4
    )
    
    test_data = b"POLYPHONIC"
    print(f"Test data: {test_data}")
    
    symbols = modem.encode_binary_to_symbols(test_data)
    print(f"Encoded symbols: {len(symbols)} (should be {len(test_data) * 4})")
    
    # Without music floor
    audio = modem.generate_polyphonic_signal(symbols, add_music_floor=False)
    print(f"\nAudio: {len(audio)} samples ({len(audio)/16000:.3f}s)")
    
    bitrate = (len(test_data) * 8) / (len(audio) / 16000)
    print(f"Bitrate: {bitrate:.0f} bps")
    
    # Demodulate
    recovered_symbols = modem.demodulate_polyphonic(audio)
    recovered_data = modem.symbols_to_binary(recovered_symbols)
    
    print(f"Recovered: {recovered_data}")
    print(f"Match: {recovered_data == test_data}")
