"""
High-Arity MFSK: Support 8-ary, 16-ary, 32-ary, 64-ary modulation.

Method: Use more distinct frequencies for higher bits-per-symbol.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class HighArityModem:
    """Supports k-ary FSK for k = 2, 4, 8, 16, 32, 64."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0, arity: int = 4):
        """
        Args:
            arity: Number of distinct frequencies (2, 4, 8, 16, 32, 64).
                   Bits per symbol = log2(arity)
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)
        self.arity = arity
        
        if arity not in [2, 4, 8, 16, 32, 64]:
            raise ValueError(f"Unsupported arity: {arity}. Use 2, 4, 8, 16, 32, or 64.")
        
        self.bits_per_symbol = int(np.log2(arity))

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to k-ary symbols."""
        symbols = []
        bits_needed = self.bits_per_symbol
        bit_buffer = 0
        bit_count = 0
        
        for byte_val in data:
            for shift in range(7, -1, -1):
                bit = (byte_val >> shift) & 1
                bit_buffer = (bit_buffer << 1) | bit
                bit_count += 1
                
                if bit_count == bits_needed:
                    symbols.append(bit_buffer)
                    bit_buffer = 0
                    bit_count = 0
        
        # Pad final symbol if needed
        if bit_count > 0:
            bit_buffer <<= (bits_needed - bit_count)
            symbols.append(bit_buffer)
        
        return np.array(symbols, dtype=np.uint8)

    def symbols_to_freqs(self, symbols: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """Map symbols to k distinct frequencies."""
        # Frequency spacing: as tight as safely possible
        # Minimum: ~50 Hz spacing for reliable detection
        freq_spacing = 100.0 if self.arity <= 8 else 75.0 if self.arity <= 16 else 50.0
        
        freq_map = {i: base_freq + i * freq_spacing for i in range(self.arity)}
        return np.array([freq_map[s] for s in symbols])

    def generate_mfsk_signal(
        self, symbols: np.ndarray, base_freq: float = 200.0, add_voice_noise: bool = True
    ) -> np.ndarray:
        """Generate k-ary FSK modulated signal."""
        freqs = self.symbols_to_freqs(symbols, base_freq)
        audio = np.array([], dtype=np.float32)

        for freq in freqs:
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            envelope = signal.get_window("hann", len(t))
            tone = np.cos(2 * np.pi * freq * t) * envelope
            audio = np.concatenate([audio, tone])

        if add_voice_noise:
            audio = self._add_voice_characteristics(audio)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def _add_voice_characteristics(self, signal_array: np.ndarray) -> np.ndarray:
        """Add voice-like harmonics."""
        t = np.arange(len(signal_array)) / self.sample_rate
        subharmonic = 0.1 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        return (signal_array + subharmonic) * am

    def demodulate_mfsk(self, audio: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """Demodulate k-ary FSK."""
        freq_spacing = 100.0 if self.arity <= 8 else 75.0 if self.arity <= 16 else 50.0
        freq_map = {i: base_freq + i * freq_spacing for i in range(self.arity)}
        
        symbols = []
        for i in range(len(audio) // self.samples_per_symbol):
            frame = audio[i * self.samples_per_symbol : (i + 1) * self.samples_per_symbol]
            
            energies = {}
            for sym, freq in freq_map.items():
                t = np.arange(len(frame)) / self.sample_rate
                carrier_cos = np.cos(2 * np.pi * freq * t)
                carrier_sin = np.sin(2 * np.pi * freq * t)
                i_comp = np.sum(frame * carrier_cos)
                q_comp = np.sum(frame * carrier_sin)
                energy = i_comp ** 2 + q_comp ** 2
                energies[sym] = energy
            
            best_sym = max(energies, key=energies.get)
            symbols.append(best_sym)
        
        return np.array(symbols, dtype=np.uint8)

    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """Convert k-ary symbols back to binary."""
        bits = []
        for sym in symbols:
            for i in range(self.bits_per_symbol - 1, -1, -1):
                bits.append((sym >> i) & 1)
        
        # Convert bits to bytes
        data = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_val = 0
                for j in range(8):
                    byte_val = (byte_val << 1) | bits[i + j]
                data.append(byte_val)
        
        return bytes(data)
