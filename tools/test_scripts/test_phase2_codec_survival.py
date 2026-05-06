#!/usr/bin/env python3
"""
Phase 2: Full Codec Survival Test with Real Loopback

Tests Phase 2 signal through real FFmpeg codecs to measure actual BER.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_detector.music_mode_modem import MusicModeModem


def test_codec_loopback(
    audio: np.ndarray,
    modem: MusicModeModem,
    test_data: bytes,
    codec: str,
    bitrate_kb: int,
) -> dict:
    """Test audio through codec loopback."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save original audio
        orig_file = tmpdir / "original.wav"
        sf.write(orig_file, audio, 16000)
        
        # Encode to codec
        encoded_file = tmpdir / f"encoded.{codec if codec != 'aac' else 'aac'}"
        if codec == "aac":
            cmd = [
                "ffmpeg", "-i", str(orig_file), "-c:a", "aac", 
                "-b:a", f"{bitrate_kb}k", "-v", "0", "-y", str(encoded_file)
            ]
        elif codec == "opus":
            cmd = [
                "ffmpeg", "-i", str(orig_file), "-c:a", "libopus",
                "-b:a", f"{bitrate_kb}k", "-v", "0", "-y", str(encoded_file)
            ]
        elif codec == "amr-nb":
            cmd = [
                "ffmpeg", "-i", str(orig_file), "-c:a", "libopencore_amrnb",
                "-b:a", f"{bitrate_kb}k", "-v", "0", "-y", str(encoded_file)
            ]
        else:
            return {"error": f"Unknown codec: {codec}"}
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return {"error": f"Encode failed: {e.stderr.decode()}"}
        
        # Decode back
        decoded_file = tmpdir / "decoded.wav"
        cmd = [
            "ffmpeg", "-i", str(encoded_file), "-ar", "16000", "-ac", "1",
            "-v", "0", "-y", str(decoded_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return {"error": f"Decode failed: {e.stderr.decode()}"}
        
        # Load decoded audio
        try:
            decoded_audio, sr = sf.read(decoded_file)
            if sr != 16000:
                raise ValueError(f"Sample rate mismatch: {sr}")
        except Exception as e:
            return {"error": f"Load failed: {e}"}
        
        # Demodulate
        try:
            symbols = modem.demodulate_chirp(decoded_audio)
            recovered_data = modem.symbols_to_binary(symbols)
        except Exception as e:
            return {"error": f"DeModulation failed: {e}"}
        
        # Calculate BER
        bits_total = len(test_data) * 8
        bits_error = 0
        
        for i, orig_byte in enumerate(test_data):
            if i < len(recovered_data):
                recov_byte = recovered_data[i]
                bits_error += bin(orig_byte ^ recov_byte).count('1')
            else:
                bits_error += 8
        
        ber = (bits_error / bits_total * 100) if bits_total > 0 else 100
        
        return {
            "ber": ber,
            "bits_error": bits_error,
            "bits_total": bits_total,
            "status": "✓ PASS" if ber < 5.0 else "✗ FAIL"
        }


def main():
    print("\n" + "=" * 100)
    print("PHASE 2: CODEC SURVIVAL TEST (Real FFmpeg Loopback)")
    print("=" * 100)
    
    test_data = b"Phase2MusicModeTest" * 5  # 95 bytes
    
    configs = [
        ("Audio with Music Floor", 2.0, 0.25, True),
        ("Audio without Music Floor", 2.0, 0.25, False),
    ]
    
    for config_label, sym_dur, overlap, add_floor in configs:
        print(f"\n[{config_label}] - Symbol {sym_dur}ms, Overlap {overlap}")
        print("-" * 100)
        
        modem = MusicModeModem(
            symbol_duration_ms=sym_dur,
            chirp_overlap=overlap
        )
        
        symbols = modem.encode_binary_to_symbols(test_data)
        audio = modem.generate_music_mode_signal(symbols, add_music_floor=add_floor)
        bitrate = (len(test_data) * 8) / (len(audio) / 16000)
        
        print(f"Payload bitrate: {bitrate:.0f} bps")
        print(f"Audio: {len(audio)} samples ({len(audio)/16000:.3f}s)")
        print(f"\n{'Codec':<12} {'Bitrate':<12} {'BER':<10} {'Errors':<10} {'Status'}")
        print("-" * 100)
        
        codecs = [
            ("AAC", "aac", 32),
            ("Opus", "opus", 32),
            ("AMR-NB", "amr-nb", 12),
        ]
        
        for codec_name, codec_type, codec_bitrate in codecs:
            result = test_codec_loopback(
                audio, modem, test_data, codec_type, codec_bitrate
            )
            
            if "error" in result:
                print(f"{codec_name:<12} {codec_bitrate:<12} {'ERROR':<10} {result['error'][:30]:<10}")
            else:
                ber = result["ber"]
                errors = result["bits_error"]
                status = result["status"]
                print(f"{codec_name:<12} {codec_bitrate:<12} {ber:<10.2f}% {errors:<10} {status}")


if __name__ == "__main__":
    main()
