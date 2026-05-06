"""
Audio Modem: Modulate binary data into voice-like signals and demodulate back.

For the Vocal Modem project, we encode data as variations in fundamental frequency,
simulating "phoneme codes" so the EVS codec treats it as speech/music.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class AudioModem:
    """Encode binary symbols into audio that survives voice codec compression."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 100.0):
        """
        Initialize the modem.

        Args:
            sample_rate: Audio sample rate (Hz), typically 16000 for telephony.
            symbol_duration_ms: Duration of each symbol in milliseconds.
                               100ms (default): Very robust, ~20 bps bitrate
                               50ms: More aggressive, ~40 bps bitrate
                               20ms: Fast, ~100 bps bitrate
                               10ms: Very fast, ~200 bps bitrate
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """
        Convert binary data (bytes) into symbols (0-3 or similar).

        Simple scheme: each byte becomes 4 x 2-bit symbols.
        Returns array of length len(data) * 4.
        """
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3  # Extract 2 bits
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)

    def symbols_to_freqs(self, symbols: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """
        Map symbols to distinct frequencies.

        For 4 symbols (0-3), we use widely-spaced frequencies to minimize interference.
        Spacing: 400 Hz (more robust than 100 Hz).
        """
        freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
        return np.array([freq_map[s] for s in symbols])

    def generate_mfsk_signal(
        self, symbols: np.ndarray, base_freq: float = 200.0, add_voice_noise: bool = True
    ) -> np.ndarray:
        """
        Generate MFSK modulated audio signal.

        Args:
            symbols: Array of symbol values (0-3).
            base_freq: Base frequency for symbol 0.
            add_voice_noise: If True, add speech-like noise to help codec treat as "voice".

        Returns:
            Audio signal as float32 array, normalized to [-1, 1].
        """
        freqs = self.symbols_to_freqs(symbols, base_freq)
        audio = np.array([], dtype=np.float32)

        for freq in freqs:
            # Generate one symbol duration at this frequency
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            # Cosine with smooth envelope to avoid clicks
            envelope = signal.get_window("hann", len(t))
            tone = np.cos(2 * np.pi * freq * t) * envelope
            audio = np.concatenate([audio, tone])

        # Optionally add voice-like harmonics to trick codec into "speech/music" mode
        if add_voice_noise:
            audio = self._add_voice_characteristics(audio)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8  # Leave headroom

        return audio.astype(np.float32)

    def _add_voice_characteristics(self, signal_array: np.ndarray) -> np.ndarray:
        """
        Add low-level voice harmonics to make codec classify as speech/music.

        Adds:
        - Sub-harmonic at ~100Hz (vocal formant suggestion)
        - Slight amplitude modulation (speech envelope)
        """
        t = np.arange(len(signal_array)) / self.sample_rate
        # Add a gentle 100Hz sub-harmonic
        subharmonic = 0.1 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
        # Add slight amplitude modulation (0.5-1.0 gain over ~1s)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        return (signal_array + subharmonic) * am

    def add_awgn(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add Additive White Gaussian Noise (AWGN) to audio.

        Args:
            audio: Input audio signal.
            snr_db: Signal-to-Noise Ratio in dB. Higher = less noise.
                   20 dB: Moderate noise
                   10 dB: Heavy noise
                   0 dB: Signal and noise power equal

        Returns:
            Audio with added Gaussian noise.
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate and add Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        noisy_audio = audio + noise
        
        # Clip to prevent overflow
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return noisy_audio.astype(np.float32)

    def demodulate_mfsk(self, audio: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """
        Demodulate MFSK signal back to symbols.

        Uses improved matched-filter detection with proper normalization.
        """
        freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
        symbols = []

        for i in range(len(audio) // self.samples_per_symbol):
            frame = audio[i * self.samples_per_symbol : (i + 1) * self.samples_per_symbol]

            # Compute correlation (matched filter) at each symbol frequency
            energies = {}
            for sym, freq in freq_map.items():
                t = np.arange(len(frame)) / self.sample_rate
                # Matched filter: correlate with cosine
                carrier_cos = np.cos(2 * np.pi * freq * t)
                carrier_sin = np.sin(2 * np.pi * freq * t)
                # Energy is I^2 + Q^2 (quadrature detection)
                i_comp = np.sum(frame * carrier_cos)
                q_comp = np.sum(frame * carrier_sin)
                energy = i_comp ** 2 + q_comp ** 2
                energies[sym] = energy

            # Pick the symbol with highest energy
            best_sym = max(energies, key=energies.get)
            symbols.append(best_sym)

        return np.array(symbols, dtype=np.uint8)

    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """
        Convert symbols back to binary data.

        Reverse of encode_binary_to_symbols: each 4 symbols = 1 byte (4 x 2-bit).
        """
        data = []
        for i in range(0, len(symbols), 4):
            if i + 4 > len(symbols):
                break  # Incomplete byte
            byte_val = 0
            for j, shift in enumerate([6, 4, 2, 0]):
                byte_val |= (symbols[i + j] & 0x3) << shift
            data.append(byte_val)
        return bytes(data)
