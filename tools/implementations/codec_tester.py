"""
Phase 1: Digital Lab - EVS/Codec Robustness Loopback Test.

Tests whether your encoded data survives voice compression by:
1. Modulating binary data into audio
2. Encoding via codec (via ffmpeg)
3. Decoding from codec
4. Demodulating back to binary
5. Measuring Bit Error Rate (BER)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_modem import AudioModem


class CodecLoopbackTester:
    """Test data survival through voice codecs (EVS, AMR-WB, etc.)."""

    # Codec command presets: map codec name to ffmpeg encoding arguments
    CODEC_PRESETS = {
        "pcm": {
            "encode_args": ["-codec:a", "pcm_s16le", "-ar", "16000"],
            "decode_args": [],
            "bitrate": None,  # PCM for baseline
            "ext": ".wav",
        },
        "aac": {
            "encode_args": ["-codec:a", "aac", "-ar", "16000", "-b:a", "32k"],
            "decode_args": [],
            "bitrate": 32000,
            "ext": ".m4a",
        },
        "opus": {
            "encode_args": ["-codec:a", "libopus", "-ar", "16000", "-b:a", "32k", "-application", "voip"],
            "decode_args": [],
            "bitrate": 32000,
            "ext": ".opus",
        },
        "amr-nb": {
            "encode_args": ["-codec:a", "libopencore_amrnb", "-ar", "8000", "-b:a", "12200"],
            "decode_args": [],
            "bitrate": 12200,
            "ext": ".amr",
        },
    }

    def __init__(self, codec: str = "pcm", sample_rate: int = 16000, symbol_duration_ms: float = 100.0):
        """
        Initialize loopback tester.

        Args:
            codec: Codec name (pcm, aac, opus, amr-nb, etc.)
            sample_rate: Target sample rate (Hz).
            symbol_duration_ms: Duration of each modem symbol in milliseconds.
        """
        self.codec = codec
        self.sample_rate = sample_rate
        self.symbol_duration_ms = symbol_duration_ms
        self.modem = AudioModem(sample_rate=sample_rate, symbol_duration_ms=symbol_duration_ms)

    def encode_audio_with_ffmpeg(self, wav_path: Path | str, bitrate_kbps: int | None = None) -> bytes:
        """
        Encode audio via ffmpeg using the selected codec.

        Returns compressed audio bytes.
        """
        if self.codec not in self.CODEC_PRESETS:
            raise ValueError(f"Unsupported codec: {self.codec}. Choose from {list(self.CODEC_PRESETS.keys())}")

        preset = self.CODEC_PRESETS[self.codec]
        codec_args = preset.get("encode_args", [])
        output_ext = preset.get("ext", ".wav")

        with tempfile.NamedTemporaryFile(suffix=output_ext, delete=False) as tmp:
            output_path = tmp.name

        try:
            cmd = (
                ["ffmpeg", "-i", str(wav_path), "-y", "-loglevel", "error"]
                + codec_args
                + [output_path]
            )
            subprocess.run(cmd, check=True, capture_output=True)
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def decode_audio_with_ffmpeg(self, encoded_bytes: bytes, bitrate_kbps: int | None = None) -> np.ndarray:
        """
        Decode compressed audio back via ffmpeg.

        Returns audio as float32 numpy array.
        """
        if self.codec not in self.CODEC_PRESETS:
            raise ValueError(f"Unsupported codec: {self.codec}")

        preset = self.CODEC_PRESETS[self.codec]
        input_ext = preset.get("ext", ".wav")

        with tempfile.NamedTemporaryFile(suffix=input_ext, delete=False) as tmp_in:
            tmp_in.write(encoded_bytes)
            tmp_in.flush()
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            cmd = [
                "ffmpeg", "-i", input_path, "-y", "-loglevel", "error", 
                "-ar", str(self.sample_rate), "-acodec", "pcm_s16le",
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            audio, sr = sf.read(output_path, dtype="float32")
            if sr != self.sample_rate:
                # Resample if needed
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate).astype(np.float32)
            return audio
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def run_loopback_test(
        self, input_data: bytes, base_freq: float = 200.0, snr_db: float | None = None
    ) -> dict:
        """
        Run full loopback test: data → modulate → encode → decode → demodulate → data.

        Args:
            input_data: Binary data (bytes) to test.
            base_freq: Base frequency for MFSK modulation (Hz).
            snr_db: Optional SNR in dB for adding AWGN before codec. None = no noise.

        Returns:
            dict with results:
            {
                'input_data': bytes (hex string in JSON),
                'output_data': bytes (hex string in JSON),
                'bit_errors': int,
                'bit_error_rate': float (0-1),
                'symbol_errors': int,
                'total_bits': int,
                'total_symbols': int,
                'codec': str,
                'sample_rate': int,
                'symbol_duration_ms': float,
                'base_freq': float,
                'snr_db': float or None,
            }
        """
        # Step 1: Modulate
        symbols = self.modem.encode_binary_to_symbols(input_data)
        audio_modulated = self.modem.generate_mfsk_signal(symbols, base_freq=base_freq, add_voice_noise=True)

        # Step 1b: Optionally add noise
        if snr_db is not None:
            audio_modulated = self.modem.add_awgn(audio_modulated, snr_db=snr_db)

        # Step 2: Write to temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = Path(tmp_wav.name)
            sf.write(wav_path, audio_modulated, self.sample_rate, subtype="PCM_16")

        try:
            # Step 3: Encode via codec
            encoded_bytes = self.encode_audio_with_ffmpeg(wav_path)

            # Step 4: Decode from codec
            audio_decoded = self.decode_audio_with_ffmpeg(encoded_bytes)

            # Normalize decoded audio
            audio_decoded = audio_decoded / (np.max(np.abs(audio_decoded)) + 1e-8) * 0.8

            # Step 5: Demodulate
            symbols_recovered = self.modem.demodulate_mfsk(audio_decoded, base_freq=base_freq)

            # Pad if needed (demodulation may lose trailing symbols)
            if len(symbols_recovered) < len(symbols):
                symbols_recovered = np.pad(symbols_recovered, (0, len(symbols) - len(symbols_recovered)))
            else:
                symbols_recovered = symbols_recovered[: len(symbols)]

            # Step 6: Convert back to binary
            output_data = self.modem.symbols_to_binary(symbols_recovered)

            # Step 7: Measure BER
            input_bits = np.unpackbits(np.frombuffer(input_data, dtype=np.uint8))
            output_bits = np.unpackbits(
                np.frombuffer(output_data, dtype=np.uint8)
            ) if output_data else np.array([])

            # Pad to same length
            max_len = max(len(input_bits), len(output_bits))
            input_bits = np.pad(input_bits, (0, max_len - len(input_bits)))
            output_bits = np.pad(output_bits, (0, max_len - len(output_bits)))

            bit_errors = np.sum(input_bits != output_bits)
            ber = bit_errors / max_len if max_len > 0 else 1.0

            # Symbol error rate
            symbol_errors = np.sum(symbols != symbols_recovered)
            total_symbols = len(symbols)

            return {
                "input_data": input_data,
                "output_data": output_data,
                "bit_errors": int(bit_errors),
                "bit_error_rate": float(ber),
                "symbol_errors": int(symbol_errors),
                "total_bits": int(max_len),
                "total_symbols": int(total_symbols),
                "codec": self.codec,
                "sample_rate": self.sample_rate,
                "symbol_duration_ms": self.symbol_duration_ms,
                "base_freq": base_freq,
                "snr_db": snr_db,
            }
        finally:
            wav_path.unlink(missing_ok=True)
