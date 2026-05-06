"""
Accordion Modulation: Multi-layer frequency encoding.

Strategy:
- Layer 1: Low band (100-500 Hz) - 4 tones
- Layer 2: Mid band (1000-2000 Hz) - 4 tones
- Layer 3: High band (3000-4000 Hz) - 4 tones
- Transmit ALL simultaneously → 3x bitrate potential

Analogy: Like a musical accordion with multiple frequency ranges.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class AccordionModem:
    """Multi-layer MFSK: encode on 3 independent frequency bands simultaneously."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)
        
        # Define frequency layers (accordion bellows)
        self.layers = {
            "low": {"base": 150, "spacing": 80, "tones": 4},      # 150, 230, 310, 390
            "mid": {"base": 1200, "spacing": 150, "tones": 4},    # 1200, 1350, 1500, 1650
            "high": {"base": 3500, "spacing": 250, "tones": 4},   # 3500, 3750, 4000, 4250
        }

    def encode_binary_to_symbols(self, data: bytes) -> dict:
        """
        Split data across 3 layers.
        Each byte → 3 symbols (one per layer).
        """
        layers_data = {"low": [], "mid": [], "high": []}
        
        for byte_val in data:
            # Split byte: bits 0-1 for low, 2-3 for mid, 4-5 for high, 6-7 unused
            low_sym = (byte_val >> 0) & 0x3
            mid_sym = (byte_val >> 2) & 0x3
            high_sym = (byte_val >> 4) & 0x3
            
            layers_data["low"].append(low_sym)
            layers_data["mid"].append(mid_sym)
            layers_data["high"].append(high_sym)
        
        return layers_data

    def generate_accordion_signal(self, layers_data: dict, add_voice_noise: bool = True) -> np.ndarray:
        """
        Generate audio with all 3 layers simultaneously.
        
        Each symbol period: low_tone + mid_tone + high_tone mixed together.
        """
        num_symbols = len(layers_data["low"])
        max_samples = num_symbols * self.samples_per_symbol
        combined_audio = np.zeros(max_samples, dtype=np.float32)
        
        for i in range(num_symbols):
            t = np.linspace(i * self.frame_duration, (i + 1) * self.frame_duration, 
                           self.samples_per_symbol, endpoint=False, dtype=np.float32)
            envelope = signal.get_window("hann", self.samples_per_symbol)
            
            # Add all three layers to the same time slot
            for layer_name in ["low", "mid", "high"]:
                symbol = layers_data[layer_name][i]
                layer_info = self.layers[layer_name]
                
                freq = layer_info["base"] + symbol * layer_info["spacing"]
                tone = np.cos(2 * np.pi * freq * t) * envelope / 3.0  # Divide by 3 to prevent clipping
                
                start_idx = i * self.samples_per_symbol
                end_idx = start_idx + self.samples_per_symbol
                combined_audio[start_idx:end_idx] += tone
        
        if add_voice_noise:
            combined_audio = self._add_voice_characteristics(combined_audio)
        
        # Normalize
        max_val = np.max(np.abs(combined_audio))
        if max_val > 0:
            combined_audio = combined_audio / max_val * 0.8
        
        return combined_audio.astype(np.float32)

    def _add_voice_characteristics(self, signal_array: np.ndarray) -> np.ndarray:
        t = np.arange(len(signal_array)) / self.sample_rate
        subharmonic = 0.05 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
        am = 0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
        return (signal_array + subharmonic) * am

    def demodulate_accordion(self, audio: np.ndarray) -> dict:
        """Demodulate 3 independent layers."""
        layers_symbols = {"low": [], "mid": [], "high": []}
        
        num_symbols = len(audio) // self.samples_per_symbol
        
        for i in range(num_symbols):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + self.samples_per_symbol
            frame = audio[start_idx:end_idx]
            
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False)
            
            # Demodulate each layer independently
            for layer_name in ["low", "mid", "high"]:
                layer_info = self.layers[layer_name]
                
                energies = {}
                for sym in range(layer_info["tones"]):
                    freq = layer_info["base"] + sym * layer_info["spacing"]
                    
                    carrier_cos = np.cos(2 * np.pi * freq * t)
                    carrier_sin = np.sin(2 * np.pi * freq * t)
                    
                    i_comp = np.sum(frame * carrier_cos)
                    q_comp = np.sum(frame * carrier_sin)
                    energy = i_comp ** 2 + q_comp ** 2
                    energies[sym] = energy
                
                best_sym = max(energies, key=energies.get)
                layers_symbols[layer_name].append(best_sym)
        
        return layers_symbols

    def symbols_to_binary(self, layers_symbols: dict) -> bytes:
        """Reconstruct bytes from 3 layers."""
        data = []
        
        for i in range(len(layers_symbols["low"])):
            byte_val = 0
            byte_val |= (layers_symbols["low"][i] & 0x3) << 0
            byte_val |= (layers_symbols["mid"][i] & 0x3) << 2
            byte_val |= (layers_symbols["high"][i] & 0x3) << 4
            data.append(byte_val)
        
        return bytes(data)


class ChirpModem:
    """
    Chirp modulation: encode data in frequency sweep direction.
    
    Instead of constant tone at freq F, use a chirp:
    - Upchirp (1000→2000 Hz): symbol 0
    - Downchirp (2000→1000 Hz): symbol 1
    - Fast upchirp: symbol 2
    - Fast downchirp: symbol 3
    
    Advantage: Less vulnerable to filter effects, more bandwidth utilization.
    """

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Simple: 2 bits per byte → 4 symbols per byte."""
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)

    def generate_chirp_signal(self, symbols: np.ndarray, add_voice_noise: bool = True) -> np.ndarray:
        """Generate chirp-modulated signal."""
        audio = np.array([], dtype=np.float32)
        
        for sym in symbols:
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)
            
            # Define chirp types
            if sym == 0:  # Upchirp 1000→2000 Hz
                freq_start, freq_end = 1000, 2000
            elif sym == 1:  # Downchirp 2000→1000 Hz
                freq_start, freq_end = 2000, 1000
            elif sym == 2:  # Fast upchirp 500→3000 Hz
                freq_start, freq_end = 500, 3000
            else:  # Fast downchirp 3000→500 Hz
                freq_start, freq_end = 3000, 500
            
            # Generate chirp: linearly swept frequency
            phase = 2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * self.frame_duration))
            envelope = signal.get_window("hann", len(t))
            chirp_tone = np.cos(phase) * envelope
            
            audio = np.concatenate([audio, chirp_tone])
        
        if add_voice_noise:
            t_all = np.arange(len(audio)) / self.sample_rate
            subharmonic = 0.08 * np.sin(2 * np.pi * 90 * t_all).astype(np.float32)
            am = 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t_all)
            audio = (audio + subharmonic) * am
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        
        return audio.astype(np.float32)

    def demodulate_chirp(self, audio: np.ndarray) -> np.ndarray:
        """Demodulate chirp signal using correlation with known chirps."""
        chirp_templates = {}
        
        # Create reference chirps for each symbol
        t_template = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False)
        
        freqs_pairs = [(1000, 2000), (2000, 1000), (500, 3000), (3000, 500)]
        
        for sym, (f_start, f_end) in enumerate(freqs_pairs):
            phase = 2 * np.pi * (f_start * t_template + (f_end - f_start) * t_template**2 / (2 * self.frame_duration))
            chirp_templates[sym] = np.cos(phase)
        
        # Correlate received signal with each template
        symbols = []
        
        for i in range(len(audio) // self.samples_per_symbol):
            frame = audio[i * self.samples_per_symbol : (i + 1) * self.samples_per_symbol]
            
            correlations = {}
            for sym, template in chirp_templates.items():
                corr = np.sum(frame * template)
                correlations[sym] = corr ** 2
            
            best_sym = max(correlations, key=correlations.get)
            symbols.append(best_sym)
        
        return np.array(symbols, dtype=np.uint8)

    def symbols_to_binary(self, symbols: np.ndarray) -> bytes:
        """Convert symbols back to bytes."""
        data = []
        for i in range(0, len(symbols), 4):
            if i + 4 <= len(symbols):
                byte_val = 0
                for j, shift in enumerate([6, 4, 2, 0]):
                    byte_val |= (symbols[i + j] & 0x3) << shift
                data.append(byte_val)
        return bytes(data)
