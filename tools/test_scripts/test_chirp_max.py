#!/usr/bin/env python3
"""
Push CHIRP modulation to its limits.

Chirp sweeps are robust because they preserve trajectory even when frequencies shift.
Let's find the maximum bitrate where chirps still survive.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess

from voice_detector.radical_modem import ChirpModem


def test_chirp_bitrate(symbol_duration_ms: float, test_data: bytes) -> dict:
    """Test chirp at specific symbol duration."""
    modem = ChirpModem(symbol_duration_ms=symbol_duration_ms)
    bitrate = (2 * 8) / (symbol_duration_ms / 1000.0)
    
    symbols = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_chirp_signal(symbols)
    
    # Write to WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
    
    try:
        # Encode with AAC
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_enc:
            enc_path = tmp_enc.name
        cmd_enc = ["ffmpeg", "-i", str(wav_path), "-codec:a", "aac", "-b:a", "32k", "-y", "-loglevel", "error", enc_path]
        subprocess.run(cmd_enc, check=True, capture_output=True)
        
        # Decode
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_dec:
            dec_path = tmp_dec.name
        cmd_dec = ["ffmpeg", "-i", enc_path, "-ar", "16000", "-acodec", "pcm_s16le", "-y", "-loglevel", "error", dec_path]
        subprocess.run(cmd_dec, check=True, capture_output=True)
        
        decoded, _ = sf.read(dec_path, dtype="float32")
        decoded = decoded / (np.max(np.abs(decoded)) + 1e-8) * 0.8
        
        # Demodulate
        symbols_recovered = modem.demodulate_chirp(decoded)
        if len(symbols_recovered) < len(symbols):
            symbols_recovered = np.pad(symbols_recovered, (0, len(symbols) - len(symbols_recovered)))
        else:
            symbols_recovered = symbols_recovered[:len(symbols)]
        
        output_data = modem.symbols_to_binary(symbols_recovered)
        
        # Calculate BER
        input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
        output_bits = np.unpackbits(np.frombuffer(output_data, dtype=np.uint8)) if output_data else np.array([])
        
        max_len = max(len(input_bits), len(output_bits))
        input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
        output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))
        
        bit_errors = np.sum(input_bits != output_bits)
        ber = bit_errors / max_len if max_len > 0 else 1.0
        
        return {
            "symbol_duration_ms": symbol_duration_ms,
            "bitrate": bitrate,
            "ber": ber,
            "bit_errors": int(bit_errors),
            "total_bits": int(max_len),
            "status": "✓" if ber <= 0.05 else "✗",
        }
    finally:
        wav_path.unlink(missing_ok=True)
        Path(enc_path).unlink(missing_ok=True)
        Path(dec_path).unlink(missing_ok=True)


def main() -> None:
    test_data = np.random.bytes(200)
    
    print("\n" + "=" * 100)
    print("CHIRP MODULATION: BITRATE SWEEP")
    print("=" * 100)
    print(f"\nChirp technique: frequency sweeps are ROBUST to codec!")
    print(f"Test data: {len(test_data)} bytes\n")
    
    print(f"{'Duration (ms)':<15} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<10}")
    print("-" * 100)
    
    durations = [100, 50, 20, 10, 5, 3, 2, 1, 0.5]
    best = None
    
    for duration_ms in durations:
        try:
            result = test_chirp_bitrate(duration_ms, test_data)
            print(
                f"{result['symbol_duration_ms']:<15} {result['bitrate']:<18.0f} "
                f"{result['ber']:<12.4f} {result['status']:<10}"
            )
            
            if result["status"] == "✓" and (best is None or result["bitrate"] > best["bitrate"]):
                best = result
        except Exception as e:
            print(f"{duration_ms:<15} ERROR: {str(e)[:50]}")
    
    if best:
        print("\n" + "=" * 100)
        print(f"🔥 BEST CHIRP RESULT: {best['symbol_duration_ms']}ms symbols")
        print(f"   Bitrate: {best['bitrate']:.0f} bps")
        print(f"   Progress toward 50 kbps: {best['bitrate']/50000*100:.2f}%")
        print(f"   BER: {best['ber']:.4f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
