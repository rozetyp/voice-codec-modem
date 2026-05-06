#!/usr/bin/env python3
"""
Test creative modulation techniques:
1. Accordion (3 simultaneous frequency layers) → 3x bitrate potential
2. Chirp modulation (frequency sweep) → robust to filtering
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess

from voice_detector.creative_modems import AccordionModem, ChirpModem


def test_accordion_codec(test_data: bytes, symbol_duration_ms: float = 10.0) -> dict:
    """Test accordion modulation through AAC codec."""
    
    modem = AccordionModem(symbol_duration_ms=symbol_duration_ms)
    bitrate = 6 * 8 / (symbol_duration_ms / 1000.0)  # 3 layers × 2 bits each
    
    # Encode
    layers_data = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_accordion_signal(layers_data, add_voice_noise=True)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)
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
        
        decoded, sr = sf.read(dec_path, dtype="float32")
        if sr != 16000:
            import librosa
            decoded = librosa.resample(decoded, orig_sr=sr, target_sr=16000).astype(np.float32)
        
        Path(enc_path).unlink(missing_ok=True)
        Path(dec_path).unlink(missing_ok=True)
        
        # Demodulate
        decoded = decoded / (np.max(np.abs(decoded)) + 1e-8) * 0.8
        layers_recovered = modem.demodulate_accordion(decoded)
        
        # Convert back
        output_data = modem.symbols_to_binary(layers_recovered)
        
        # Calculate BER
        input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
        output_bits = np.unpackbits(np.frombuffer(output_data, dtype=np.uint8)) if output_data else np.array([])
        
        max_len = max(len(input_bits), len(output_bits))
        input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
        output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))
        
        bit_errors = np.sum(input_bits != output_bits)
        ber = bit_errors / max_len if max_len > 0 else 1.0
        
        return {
            "technique": "Accordion",
            "symbol_duration_ms": symbol_duration_ms,
            "bitrate": bitrate,
            "bit_errors": int(bit_errors),
            "bit_error_rate": float(ber),
            "total_bits": int(max_len),
            "status": "✓" if ber <= 0.05 else "✗",
        }
    finally:
        wav_path.unlink(missing_ok=True)


def test_chirp_codec(test_data: bytes, symbol_duration_ms: float = 10.0) -> dict:
    """Test chirp modulation through AAC codec."""
    
    modem = ChirpModem(symbol_duration_ms=symbol_duration_ms)
    bitrate = 2 * 8 / (symbol_duration_ms / 1000.0)  # 2 bits per symbol
    
    # Encode
    symbols = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_chirp_signal(symbols, add_voice_noise=True)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)
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
        
        decoded, sr = sf.read(dec_path, dtype="float32")
        if sr != 16000:
            import librosa
            decoded = librosa.resample(decoded, orig_sr=sr, target_sr=16000).astype(np.float32)
        
        Path(enc_path).unlink(missing_ok=True)
        Path(dec_path).unlink(missing_ok=True)
        
        # Demodulate
        decoded = decoded / (np.max(np.abs(decoded)) + 1e-8) * 0.8
        symbols_recovered = modem.demodulate_chirp(decoded)
        
        # Convert back
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
            "technique": "Chirp",
            "symbol_duration_ms": symbol_duration_ms,
            "bitrate": bitrate,
            "bit_errors": int(bit_errors),
            "bit_error_rate": float(ber),
            "total_bits": int(max_len),
            "status": "✓" if ber <= 0.05 else "✗",
        }
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> None:
    test_data = np.random.bytes(200)
    
    print("\n" + "=" * 100)
    print("CREATIVE MODULATION TECHNIQUES")
    print("=" * 100)
    print(f"\nTest data: {len(test_data)} bytes, Codec: AAC\n")
    
    print(f"{'Technique':<15} {'Symbol (ms)':<15} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<8}")
    print("-" * 100)
    
    configs = [
        ("accordion", [10, 5, 3]),
        ("chirp", [10, 5, 3, 2]),
    ]
    
    best_result = None
    
    for technique, durations in configs:
        for symbol_ms in durations:
            try:
                if technique == "accordion":
                    result = test_accordion_codec(test_data, symbol_duration_ms=symbol_ms)
                else:
                    result = test_chirp_codec(test_data, symbol_duration_ms=symbol_ms)
                
                print(
                    f"{result['technique']:<15} {result['symbol_duration_ms']:<15} "
                    f"{result['bitrate']:<18.0f} {result['bit_error_rate']:<12.4f} {result['status']:<8}"
                )
                
                if result["status"] == "✓" and (best_result is None or result["bitrate"] > best_result["bitrate"]):
                    best_result = result
            except Exception as e:
                print(f"{technique:<15} {symbol_ms:<15} ERROR: {str(e)[:50]}")
    
    if best_result:
        print("\n" + "=" * 100)
        print(f"🎯 BEST: {best_result['technique']} at {best_result['symbol_duration_ms']}ms")
        print(f"   Bitrate: {best_result['bitrate']:.0f} bps ({best_result['bitrate']/50000*100:.1f}% of 50 kbps)")
        print(f"   BER: {best_result['bit_error_rate']:.4f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
