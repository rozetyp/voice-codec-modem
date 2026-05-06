#!/usr/bin/env python3
"""
Hybrid Attack: Chirp + Overlapping
Double the bitrate by overlapping chirps while using chirp's robustness.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess
from scipy.signal import get_window


class HybridChirpModem:
    """Overlapping chirps: each new chirp starts before previous ends."""

    def __init__(self, sample_rate: int = 16000, symbol_duration_ms: float = 5.0, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.frame_duration = symbol_duration_ms / 1000.0
        self.samples_per_symbol = int(self.sample_rate * self.frame_duration)
        self.overlap = overlap
        self.step_samples = int(self.samples_per_symbol * (1.0 - overlap))

    def encode_binary_to_symbols(self, data: bytes) -> np.ndarray:
        """Convert bytes to 4-ary symbols."""
        symbols = []
        for byte_val in data:
            for shift in [6, 4, 2, 0]:
                sym = (byte_val >> shift) & 0x3
                symbols.append(sym)
        return np.array(symbols, dtype=np.uint8)

    def generate_hybrid_signal(self, symbols: np.ndarray) -> np.ndarray:
        """
        Generate overlapping chirps.
        Symbols start every step_samples instead of every samples_per_symbol.
        """
        num_symbols = len(symbols)
        total_samples = self.samples_per_symbol + (num_symbols - 1) * self.step_samples
        audio = np.zeros(total_samples, dtype=np.float32)

        for idx, sym in enumerate(symbols):
            # Chirp band for this symbol
            sym_int = int(sym) % 4
            f0 = 200.0 + sym_int * 1000.0
            f1 = f0 + 1000.0

            # Generate chirp
            start_sample = idx * self.step_samples
            t = np.linspace(0, self.frame_duration, self.samples_per_symbol, endpoint=False, dtype=np.float32)

            # Linear chirp
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.frame_duration))
            chirp = np.cos(phase)

            # Envelope
            envelope = get_window("hann", len(t))
            chirp = chirp * envelope

            # Add to output with overlap
            end_sample = min(start_sample + self.samples_per_symbol, total_samples)
            audio[start_sample:end_sample] += chirp[:end_sample - start_sample]

        # Normalize
        max_val = np.max(np.abs(audio)) + 1e-8
        audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def demodulate_hybrid(self, audio: np.ndarray) -> np.ndarray:
        """Extract overlapping chirps."""
        symbols = []
        num_symbols = (len(audio) - self.samples_per_symbol) // self.step_samples + 1

        for idx in range(num_symbols):
            start_sample = idx * self.step_samples
            end_sample = min(start_sample + self.samples_per_symbol, len(audio))
            frame = audio[start_sample:end_sample]

            if len(frame) < self.samples_per_symbol // 2:
                break

            if len(frame) < self.samples_per_symbol:
                frame = np.pad(frame, (0, self.samples_per_symbol - len(frame)))

            # Find peak frequency in this frame
            fft_vals = np.fft.rfft(frame)
            power = np.abs(fft_vals) ** 2
            freq_bins = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)

            # Find band with most energy
            best_sym = 0
            best_energy = 0
            for sym in range(4):
                sym_int = int(sym) % 4
                f0 = 200.0 + sym_int * 1000.0
                f1 = f0 + 1000.0
                mask = (freq_bins >= f0) & (freq_bins < f1)
                energy = np.sum(power[mask]) if np.any(mask) else 0

                if energy > best_energy:
                    best_energy = energy
                    best_sym = sym

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


def test_hybrid(overlap: float, symbol_ms: float, test_data: bytes) -> dict:
    """Test hybrid at specific parameters."""
    modem = HybridChirpModem(symbol_duration_ms=symbol_ms, overlap=overlap)
    
    # Effective bitrate (accounting for overlap)
    effective_bitrate = (2 * 8) / (modem.step_samples / modem.sample_rate)
    
    symbols = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_hybrid_signal(symbols)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
    
    try:
        # Encode/decode with AAC
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_enc:
            enc_path = tmp_enc.name
        cmd_enc = ["ffmpeg", "-i", str(wav_path), "-codec:a", "aac", "-b:a", "32k", "-y", "-loglevel", "error", enc_path]
        subprocess.run(cmd_enc, check=True, capture_output=True)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_dec:
            dec_path = tmp_dec.name
        cmd_dec = ["ffmpeg", "-i", enc_path, "-ar", "16000", "-acodec", "pcm_s16le", "-y", "-loglevel", "error", dec_path]
        subprocess.run(cmd_dec, check=True, capture_output=True)
        
        decoded, _ = sf.read(dec_path, dtype="float32")
        decoded = decoded / (np.max(np.abs(decoded)) + 1e-8) * 0.8
        
        symbols_recovered = modem.demodulate_hybrid(decoded)
        if len(symbols_recovered) < len(symbols):
            symbols_recovered = np.pad(symbols_recovered, (0, len(symbols) - len(symbols_recovered)))
        else:
            symbols_recovered = symbols_recovered[:len(symbols)]
        
        output_data = modem.symbols_to_binary(symbols_recovered)
        
        input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
        output_bits = np.unpackbits(np.frombuffer(output_data, dtype=np.uint8)) if output_data else np.array([])
        
        max_len = max(len(input_bits), len(output_bits))
        input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
        output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))
        
        bit_errors = np.sum(input_bits != output_bits)
        ber = bit_errors / max_len if max_len > 0 else 1.0
        
        return {
            "overlap": overlap,
            "symbol_ms": symbol_ms,
            "effective_bitrate": effective_bitrate,
            "ber": ber,
            "status": "✓" if ber <= 0.05 else "✗",
        }
    finally:
        wav_path.unlink(missing_ok=True)
        Path(enc_path).unlink(missing_ok=True)
        Path(dec_path).unlink(missing_ok=True)


def main() -> None:
    test_data = np.random.bytes(200)
    
    print("\n" + "=" * 100)
    print("HYBRID CHIRP + OVERLAPPING")
    print("=" * 100)
    print(f"\nUsing overlapping chirps to increase effective bitrate")
    print(f"Test data: {len(test_data)} bytes\n")
    
    print(f"{'Overlap':<10} {'Symbol (ms)':<15} {'Effective BR (bps)':<20} {'BER':<12} {'Status':<10}")
    print("-" * 100)
    
    configs = [
        (0.0, 2),
        (0.25, 2),
        (0.5, 2),
        (0.75, 2),
        (0.9, 2),
        (0.5, 1),
        (0.5, 0.5),
    ]
    
    best = None
    
    for overlap, symbol_ms in configs:
        try:
            result = test_hybrid(overlap, symbol_ms, test_data)
            print(
                f"{result['overlap']:<10} {result['symbol_ms']:<15} "
                f"{result['effective_bitrate']:<20.0f} {result['ber']:<12.4f} {result['status']:<10}"
            )
            
            if result["status"] == "✓" and (best is None or result["effective_bitrate"] > best["effective_bitrate"]):
                best = result
        except Exception as e:
            print(f"{overlap:<10} {symbol_ms:<15} ERROR: {str(e)[:50]}")
    
    if best:
        print("\n" + "=" * 100)
        print(f"🎯 BEST HYBRID: {best['overlap']*100:.0f}% overlap, {best['symbol_ms']}ms symbols")
        print(f"   Effective bitrate: {best['effective_bitrate']:.0f} bps")
        print(f"   Progress toward 50 kbps: {best['effective_bitrate']/50000*100:.2f}%")
        print(f"   BER: {best['ber']:.4f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
