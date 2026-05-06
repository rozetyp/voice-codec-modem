"""
Revolutionary Modem Techniques:
1. CHIRP/SWEEP - frequency sweeps instead of fixed tones
2. OVERLAPPING SYMBOLS - interleave transmission in time-frequency
3. FREQUENCY GRADIENT - "Accordeon" - sweep band-width dynamically
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class ChirpModem:
    """Chirp-based modulation: encode data into frequency sweeps."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to 4-ary symbols."""
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)

    def generate_chirp_signal(
        self, symbols: np.ndarray, base_freq: float = 200.0, max_freq: float = 4000.0
    ) -> np.ndarray:
        """
        Generate chirp sweep for each symbol.
        
        Symbol 0: sweep from base_freq to base_freq+1000
        Symbol 1: sweep from base_freq+1000 to base_freq+2000
        Symbol 2: sweep from base_freq+2000 to base_freq+3000
        Symbol 3: sweep from base_freq+3000 to base_freq+4000
        
        Chirps are ROBUST to codec because they preserve frequency trajectory,
        even if exact frequencies get shifted/smeared.
        """
        audio = np.array([], dtype=np.float32)
        band_width = (max_freq - base_freq) / 4.0  # 4 symbols = 4 bands

        for sym in symbols:
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            
            # Chirp band for this symbol
            f0 = base_freq + sym * band_width
            f1 = f0 + band_width
            
            # Linear chirp from f0 to f1
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.frame_duration))
            chirp = np.cos(phase)
            
            # Envelope
            envelope = signal.get_window("hann", len(t))
            chirp = chirp * envelope
            
            audio = np.concatenate([audio, chirp])

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def demodulate_chirp(self, audio: np.ndarray, base_freq: float = 200.0, max_freq: float = 4000.0) -> np.ndarray:
        """
        Demodulate chirp signal by measuring energy in each band over time.
        
        Use spectrogram to find which frequency band has most energy during each symbol period.
        """
        symbols = []
        band_width = (max_freq - base_freq) / 4.0

        for i in range(len(audio) // self.samples_per_symbol):
            frame = audio[i * self.samples_per_symbol : (i + 1) * self.samples_per_symbol]

            # Compute power spectrum
            freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)
            spectrum = np.abs(np.fft.rfft(frame)) ** 2

            # Find which band has most energy
            band_energies = []
            for sym in range(4):
                f0 = base_freq + sym * band_width
                f1 = f0 + band_width
                mask = (freqs >= f0) & (freqs < f1)
                energy = np.sum(spectrum[mask])
                band_energies.append(energy)

            best_sym = int(np.argmax(band_energies))
            symbols.append(best_sym)

        return np.array(symbols, dtype=np.uint8)

    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """Convert symbols back to binary."""
        data = []
        for i in range(0, len(symbols), 4):
            if i + 4 > len(symbols):
                break
            byte_val = 0
            for j, shift in enumerate([6, 4, 2, 0]):
                byte_val |= (symbols[i + j] & 0x3) << shift
            data.append(byte_val)
        return bytes(data)


class LayeredModem:
    """
    Overlapping/Layered modulation: transmit multiple "layers" simultaneously.
    
    Instead of: sym1 (0-T), sym2 (T-2T), sym3 (2T-3T)...
    Use: sym1 (0-T), sym2 (T/2-3T/2), sym3 (T-2T)...
    
    Increases symbol rate by overlapping them.
    Like pages in an accordion expanding/contracting.
    """

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0, overlap: float = 0.5):
        """
        overlap: 0.0 = no overlap, 0.5 = 50% overlap, 1.0 = complete overlap
        """
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)
        self.overlap = overlap  # 0.5 default = symbols start every 5ms (not every 10ms)
        self.step_samples = int(self.samples_per_symbol * (1.0 - overlap))

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to 4-ary symbols."""
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)

    def generate_layered_signal(self, symbols: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """
        Generate overlapping symbols.
        
        Each symbol is 10ms, but new symbols start every 5ms (50% overlap).
        This packs 2x more symbols in same time = 2x bitrate!
        """
        freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}

        # Calculate total length
        num_symbols = len(symbols)
        total_samples = self.samples_per_symbol + (num_symbols - 1) * self.step_samples

        audio = np.zeros(total_samples, dtype=np.float32)

        for idx, sym in enumerate(symbols):
            freq = freq_map[sym]
            start_sample = idx * self.step_samples

            # Generate symbol
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            from scipy.signal import get_window
            envelope = get_window("hann", len(t))
            tone = np.cos(2 * np.pi * freq * t) * envelope

            # Add to output (overlapping with previous)
            end_sample = min(start_sample + self.samples_per_symbol, total_samples)
            audio[start_sample:end_sample] += tone[:end_sample - start_sample]

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def demodulate_layered(self, audio: np.ndarray, base_freq: float = 200.0) -> np.ndarray:
        """
        Demodulate overlapping symbols.
        
        Extract each symbol at its start time and demodulate independently.
        """
        freq_map = {0: base_freq, 1: base_freq + 400, 2: base_freq + 800, 3: base_freq + 1200}
        symbols = []

        num_symbols = (len(audio) - self.samples_per_symbol) // self.step_samples + 1

        for idx in range(num_symbols):
            start_sample = idx * self.step_samples
            end_sample = min(start_sample + self.samples_per_symbol, len(audio))
            frame = audio[start_sample:end_sample]

            if len(frame) < self.samples_per_symbol // 2:  # Too short to decode
                break

            # Pad if needed
            if len(frame) < self.samples_per_symbol:
                frame = np.pad(frame, (0, self.samples_per_symbol - len(frame)))

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
        """Convert symbols back to binary."""
        data = []
        for i in range(0, len(symbols), 4):
            if i + 4 > len(symbols):
                break
            byte_val = 0
            for j, shift in enumerate([6, 4, 2, 0]):
                byte_val |= (symbols[i + j] & 0x3) << shift
            data.append(byte_val)
        return bytes(data)


class FrequencyGradientModem:
    """
    "Accordeon" technique: dynamically compress/expand frequency usage.
    
    Instead of fixed frequencies, use a continuous spectrum.
    Encode data as position along frequency axis: 0Hz = 00, 4000Hz = 11, etc.
    
    Continuous gradient is MORE robust to codec than discrete tones!
    """

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to continuous frequency values (0-4000)."""
        freqs = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                bits = (byte_val >> shift) & 0x3
                # Map 2 bits (0-3) to frequency (0-4000)
                freq = 200 + (bits / 3.0) * 3800
                freqs.append(freq)
        return np.array(freqs, dtype=np.float32)

    def generate_gradient_signal(self, freqs: np.ndarray) -> np.ndarray:
        """Generate continuous frequency-modulated signal."""
        audio = np.array([], dtype=np.float32)

        for target_freq in freqs:
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            
            # Generate smooth tone at target frequency
            # Use raised cosine to prevent clicks
            from scipy.signal import get_window
            envelope = get_window("hamming", len(t))
            
            tone = np.cos(2 * np.pi * target_freq * t) * envelope
            audio = np.concatenate([audio, tone])

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def demodulate_gradient(self, audio: np.ndarray) -> np.ndarray:
        """
        Demodulate by finding peak frequency in each symbol period.
        """
        freqs = []

        for i in range(len(audio) // self.samples_per_symbol):
            frame = audio[i * self.samples_per_symbol : (i + 1) * self.samples_per_symbol]

            # Compute FFT
            fft_vals = np.fft.rfft(frame)
            power = np.abs(fft_vals) ** 2
            freq_bins = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)

            # Find peak frequency
            peak_bin = int(np.argmax(power))
            if peak_bin < len(freq_bins):
                peak_freq = freq_bins[peak_bin]
            else:
                peak_freq = 200.0

            freqs.append(peak_freq)

        return np.array(freqs, dtype=np.float32)

    def freqs_to_binary(self, freqs: np.ndarray) -> bytes:
        """Convert continuous frequencies back to binary."""
        data = []
        bits_buffer = []

        for freq in freqs:
            # Map frequency (200-4000) back to 2 bits (0-3)
            normalized = np.clip((freq - 200) / 3800.0, 0, 1)
            bits = int(normalized * 3)
            bits_buffer.append(bits)

        # Convert 2-bit groups to bytes
        for i in range(0, len(bits_buffer), 4):
            if i + 4 > len(bits_buffer):
                break
            byte_val = 0
            for j, shift in enumerate([6, 4, 2, 0]):
                byte_val |= (bits_buffer[i + j] & 0x3) << shift
            data.append(byte_val)

        return bytes(data)
