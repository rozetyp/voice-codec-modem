"""
Aggressive Bitrate Push: 8-ary modulation + ultra-short symbols.

This pushes toward 50 kbps by:
1. Using 8 frequencies (3 bits/symbol) instead of 4 (2 bits/symbol)
2. Testing 1ms, 2ms, 5ms symbols (100x shorter than baseline)
3. Finding the breaking point where BER spikes
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class AudioModemAggressive:
    """High-bitrate variant: 8-ary MFSK with ultra-short symbols."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0, arity: int = 8):
        """
        Initialize aggressive modem.

        Args:
            sample_rate: 16000 Hz (telephony standard)
            symbol_duration_ms: Symbol duration (1ms = ultra-aggressive, 5ms/10ms = moderate)
            arity: 4 (2 bits), 8 (3 bits), or 16 (4 bits) - higher = faster but harder to detect
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)
        self.arity = arity
        
        if arity == 4:
            self.bits_per_symbol = 2
        elif arity == 8:
            self.bits_per_symbol = 3
        elif arity == 16:
            self.bits_per_symbol = 4
        else:
            raise ValueError(f"Unsupported arity: {arity}")

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert binary data to symbols (4-ary, 8-ary, or 16-ary)."""
        symbols = []
        bits_buffer = 0
        bits_count = 0

        for byte_val in data:
            bits_buffer = (bits_buffer << 8) | byte_val
            bits_count += 8

            while bits_count >= self.bits_per_symbol:
                bits_count -= self.bits_per_symbol
                shift = bits_count
                sym = (bits_buffer >> shift) & (self.arity - 1)
                symbols.append(sym)

        # Flush remaining bits (zero-padded)
        if bits_count > 0:
            sym = (bits_buffer << (self.bits_per_symbol - bits_count)) & (self.arity - 1)
            symbols.append(sym)

        return np.array(symbols, dtype=np.uint8)

    def symbols_to_freqs(self, symbols: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """
        Map symbols to distinct frequencies.

        For 8-ary: 200Hz, 600Hz, 1000Hz, 1400Hz, 1800Hz, 2200Hz, 2600Hz, 3000Hz
        Wider spacing for arity=8 (600 Hz apart)
        """
        if self.arity == 4:
            freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
        elif self.arity == 8:
            freq_map = {
                0: base_freq,
                1: base_freq + 600,
                2: base_freq + 1200,
                3: base_freq + 1800,
                4: base_freq + 2400,
                5: base_freq + 3000,
                6: base_freq + 3600,
                7: base_freq + 4200,
            }
        else:  # arity == 16
            freq_map = {i: base_freq + (i * 600) for i in range(16)}

        return np.array([freq_map[s] for s in symbols])

    def generate_mfsk_signal(
        self, symbols: np.ndarray, base_freq: float = 200.0, add_voice_noise: bool = True
    ) -> np.ndarray:
        """Generate MFSK signal at selected frequencies."""
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
        """Add speech-like characteristics to fool codec."""
        t = np.arange(len(signal_array)) / self.sample_rate
        subharmonic = 0.05 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
        am = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        return (signal_array + subharmonic) * am

    def demodulate_mfsk(self, audio: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """Demodulate MFSK using quadrature matched filters."""
        if self.arity == 4:
            freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
        elif self.arity == 8:
            freq_map = {
                0: base_freq,
                1: base_freq + 600,
                2: base_freq + 1200,
                3: base_freq + 1800,
                4: base_freq + 2400,
                5: base_freq + 3000,
                6: base_freq + 3600,
                7: base_freq + 4200,
            }
        else:  # arity == 16
            freq_map = {i: base_freq + (i * 600) for i in range(16)}

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
        """Convert symbols back to bytes."""
        data = []
        bits_buffer = 0
        bits_count = 0

        for sym in symbols:
            bits_buffer = (bits_buffer << self.bits_per_symbol) | (sym & (self.arity - 1))
            bits_count += self.bits_per_symbol

            while bits_count >= 8:
                bits_count -= 8
                byte_val = (bits_buffer >> bits_count) & 0xFF
                data.append(byte_val)

        return bytes(data)

    def add_awgn(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add AWGN for realistic channel simulation."""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        noisy_audio = audio + noise
        return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)
