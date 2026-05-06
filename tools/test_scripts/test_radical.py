#!/usr/bin/env python3
"""
Test radical modulation techniques against codec.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess

from voice_detector.radical_modem import ChirpModem, LayeredModem, FrequencyGradientModem


def test_modem_codec(modem_class, test_data: bytes, codec: str = "aac", **modem_kwargs) -> dict:
    """Generic test for any modem type."""
    modem = modem_class(**modem_kwargs)
    
    # Encode
    if modem_class == FrequencyGradientModem:
        freqs = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_gradient_signal(freqs)
    else:
        symbols = modem.encode_binary_to_symbols(test_data)
        if modem_class == ChirpModem:
            audio = modem.generate_chirp_signal(symbols)
        else:  # LayeredModem
            audio = modem.generate_layered_signal(symbols)
    
    # Write to WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
    
    try:
        if codec == "pcm":
            decoded = audio
        else:
            # Encode
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
            Path(enc_path).unlink(missing_ok=True)
            Path(dec_path).unlink(missing_ok=True)
        
        # Demodulate
        decoded = decoded / (np.max(np.abs(decoded)) + 1e-8) * 0.8
        
        if modem_class == ChirpModem:
            symbols_recovered = modem.demodulate_chirp(decoded)
            output_data = modem.symbols_to_binary(symbols_recovered)
        elif modem_class == LayeredModem:
            symbols_recovered = modem.demodulate_layered(decoded)
            output_data = modem.symbols_to_binary(symbols_recovered)
        else:  # FrequencyGradientModem
            freqs_recovered = modem.demodulate_gradient(decoded)
            output_data = modem.freqs_to_binary(freqs_recovered)
        
        # Calculate BER
        input_bits = np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))
        output_bits = np.unpackbits(np.frombuffer(output_data, dtype=np.uint8)) if output_data else np.array([])
        
        max_len = max(len(input_bits), len(output_bits))
        input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
        output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))
        
        bit_errors = np.sum(input_bits != output_bits)
        ber = bit_errors / max_len if max_len > 0 else 1.0
        
        return {
            "ber": ber,
            "bit_errors": int(bit_errors),
            "total_bits": int(max_len),
            "status": "✓" if ber <= 0.05 else "✗",
        }
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> None:
    test_data = np.random.bytes(150)
    
    print("\n" + "=" * 100)
    print("RADICAL MODULATION TECHNIQUES")
    print("=" * 100)
    
    techniques = [
        ("TRADITIONAL\n(Fixed 4-ary, 10ms)", lambda: None, {}),  # Baseline
        ("CHIRP\n(Frequency sweeps)", ChirpModem, {"symbol_duration_ms": 10.0}),
        ("LAYERED\n(50% overlap)", LayeredModem, {"symbol_duration_ms": 10.0, "overlap": 0.5}),
        ("LAYERED\n(75% overlap)", LayeredModem, {"symbol_duration_ms": 10.0, "overlap": 0.75}),
        ("GRADIENT\n(Continuous freq)", FrequencyGradientModem, {"symbol_duration_ms": 10.0}),
    ]
    
    print(f"\n{'Technique':<25} {'BER':<12} {'Bit Errors':<15} {'Status':<10}")
    print("-" * 100)
    
    for name, modem_class, kwargs in techniques:
        if modem_class is None:
            # Baseline: use traditional 4-ary
            from voice_detector.audio_modem import AudioModem
            modem = AudioModem(symbol_duration_ms=10.0)
            symbols = modem.encode_binary_to_symbols(test_data)
            audio = modem.generate_mfsk_signal(symbols)
            
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
                
                symbols_recovered = modem.demodulate_mfsk(decoded)
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
                
                status = "✓" if ber <= 0.05 else "✗"
                print(f"{name:<25} {ber:<12.4f} {bit_errors:<15} {status:<10}")
                
                Path(enc_path).unlink(missing_ok=True)
                Path(dec_path).unlink(missing_ok=True)
            finally:
                wav_path.unlink(missing_ok=True)
        else:
            try:
                result = test_modem_codec(modem_class, test_data, codec="aac", **kwargs)
                print(f"{name:<25} {result['ber']:<12.4f} {result['bit_errors']:<15} {result['status']:<10}")
            except Exception as e:
                print(f"{name:<25} ERROR: {str(e)[:40]}")
    
    print("\n" + "=" * 100)
    print("Interpretation:")
    print("  ✓ = BER < 5% (acceptable)")
    print("  ✗ = BER > 5% (failed)")
    print("\nLayered with overlap: theoretically 2x bitrate")
    print("Chirp: sweeps are more robust to codec distortion")
    print("Gradient: continuous freq space more preservation by codec")
    print("=" * 100)


if __name__ == "__main__":
    main()
