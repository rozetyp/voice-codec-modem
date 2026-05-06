#!/usr/bin/env python3
"""
Test high-arity modulation (8-ary, 16-ary, 32-ary).

Goal: Get as close to 50 kbps as possible by maximizing bits per symbol.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess

from voice_detector.high_arity_modem import HighArityModem


def test_high_arity_codec(
    codec: str,
    test_data: bytes,
    arity: int = 8,
    symbol_duration_ms: float = 5.0,
) -> dict:
    """Test codec survival with high-arity modulation."""
    
    modem = HighArityModem(symbol_duration_ms=symbol_duration_ms, arity=arity)
    bits_per_symbol = modem.bits_per_symbol
    bitrate = (bits_per_symbol * 8) / (symbol_duration_ms / 1000.0)
    
    # Encode
    symbols = modem.encode_binary_to_symbols(test_data)
    audio = modem.generate_mfsk_signal(symbols, base_freq=200.0, add_voice_noise=True)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
    
    try:
        # Encode with codec
        if codec == "pcm":
            encoded = audio.astype(np.float32)
            decoded = encoded
        else:
            # Use ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_enc:
                enc_path = tmp_enc.name
            
            cmd_enc = ["ffmpeg", "-i", str(wav_path), "-codec:a", codec, "-b:a", "32k", "-y", "-loglevel", "error", enc_path]
            subprocess.run(cmd_enc, check=True, capture_output=True)
            
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
        symbols_recovered = modem.demodulate_mfsk(decoded, base_freq=200.0)
        
        if len(symbols_recovered) < len(symbols):
            symbols_recovered = np.pad(symbols_recovered, (0, len(symbols) - len(symbols_recovered)))
        else:
            symbols_recovered = symbols_recovered[:len(symbols)]
        
        # Convert back to binary
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
            "arity": arity,
            "bits_per_symbol": bits_per_symbol,
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
    print("HIGH-ARITY MODULATION TESTING: 8-ary, 16-ary, 32-ary")
    print("=" * 100)
    print(f"\nTest data: {len(test_data)} bytes, Codec: AAC\n")
    
    print(f"{'Arity':<8} {'Bits/Sym':<12} {'Symbol (ms)':<15} {'Bitrate (bps)':<18} {'BER':<12} {'Status':<8}")
    print("-" * 100)
    
    configs = [
        (8, 5),
        (8, 3),
        (8, 2),
        (16, 5),
        (16, 3),
        (16, 2),
        (16, 1),
        (32, 5),
        (32, 3),
        (32, 2),
    ]
    
    best_result = None
    
    for arity, symbol_ms in configs:
        try:
            result = test_high_arity_codec("aac", test_data, arity=arity, symbol_duration_ms=symbol_ms)
            print(
                f"{result['arity']:<8} {result['bits_per_symbol']:<12} "
                f"{result['symbol_duration_ms']:<15} {result['bitrate']:<18.0f} "
                f"{result['bit_error_rate']:<12.4f} {result['status']:<8}"
            )
            
            if result["status"] == "✓" and (best_result is None or result["bitrate"] > best_result["bitrate"]):
                best_result = result
        except Exception as e:
            print(f"{arity:<8} ERROR: {str(e)[:70]}")
    
    if best_result:
        print("\n" + "=" * 100)
        print(f"🎯 BEST WORKING: {best_result['arity']}-ary, {best_result['symbol_duration_ms']}ms symbols")
        print(f"   Bitrate: {best_result['bitrate']:.0f} bps ({best_result['bitrate']/50000*100:.1f}% of 50 kbps target)")
        print(f"   BER: {best_result['bit_error_rate']:.4f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
