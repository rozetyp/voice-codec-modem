"""
Inference-only wrappers around the three trained codecs.

Three pipelines:
  - opus_iid    : ckpt_n128_mixed.pt    (3196 bps reliable through Opus 24k VOIP)
  - stego_opus  : ckpt_stego_p3.pt      (270 bps reliable, real-speech cover, 24k VOIP)
  - stego_amrnb : ckpt_amrnb_real.pt    (~76 bps reliable, AMR-NB cellular voice)

Each exposes encode_text(text)->wav_bytes and decode_audio(wav_bytes)->dict.

Model classes copied here (vs imported from core/) so this module is self-contained
and doesn't drag training-time dependencies into the deployed image.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import reedsolo
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- constants ----------
SR = 16_000
SYMBOL_MS = 30
SYMBOL_N = SR * SYMBOL_MS // 1000  # 480 samples per IID/stego symbol
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# Repo paths (resolved relative to this file so Railway and local work the same)
ROOT = Path(__file__).resolve().parent.parent
CKPT_IID  = ROOT / "core" / "neural_codec" / "ckpt_n128_mixed.pt"
CKPT_STEG = ROOT / "core" / "neural_codec" / "stego" / "ckpt_stego_p3.pt"
CKPT_AMR  = ROOT / "core" / "neural_codec" / "cellular" / "ckpt_amrnb_real.pt"
COVER_AUDIO = ROOT / "app" / "static" / "cover.wav"  # 30 sec real speech for stego


# ---------- model classes (copied from core/, inference-only) ----------
class Encoder(nn.Module):
    """bits (B, n_bits) -> audio (B, SYMBOL_N) in [-1, 1]."""
    def __init__(self, n_bits: int, hidden: int = 192, base_len: int = 60):
        super().__init__()
        self.n_bits = n_bits
        self.base_len = base_len
        self.fc = nn.Linear(n_bits, hidden * base_len)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(hidden, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),     nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),      nn.GELU(),
            nn.Conv1d(32, 1, 7, 1, 3),                nn.Tanh(),
        )

    def forward(self, bits):
        x = self.fc(bits.float() * 2 - 1)
        x = x.view(x.size(0), -1, self.base_len)
        x = self.up(x).squeeze(1)
        ramp = max(8, SYMBOL_N // 60)
        win = torch.ones(SYMBOL_N, device=x.device)
        win[:ramp] = torch.linspace(0, 1, ramp, device=x.device)
        win[-ramp:] = torch.linspace(1, 0, ramp, device=x.device)
        return x * win


class Decoder(nn.Module):
    def __init__(self, n_bits: int, hidden: int = 192):
        super().__init__()
        self.n_bits = n_bits
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3),  nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1),  nn.GELU(),
            nn.Conv1d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv1d(128, hidden, 4, 2, 1), nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 60, 256), nn.GELU(),
            nn.Linear(256, n_bits),
        )

    def forward(self, audio):
        x = audio.unsqueeze(1)
        x = self.conv(x).flatten(1)
        return self.fc(x)


class StegEncoder(nn.Module):
    """(cover_audio, bits) -> modified_audio."""
    def __init__(self, n_bits: int, hidden: int = 96):
        super().__init__()
        self.n_bits = n_bits
        self.bit_fc = nn.Linear(n_bits, hidden * 60)
        self.cover_enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv1d(64, hidden, 4, 2, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 4, 2, 1), nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(hidden, 64, 4, 2, 1),       nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),           nn.GELU(),
            nn.Conv1d(32, 1, 7, 1, 3),                     nn.Tanh(),
        )

    def forward(self, cover, bits, perturbation_scale: float = 0.5):
        c_feat = self.cover_enc(cover.unsqueeze(1))
        b_feat = self.bit_fc(bits.float() * 2 - 1).view(c_feat.size(0), -1, 60)
        joint = torch.cat([c_feat, b_feat], dim=1)
        delta = self.dec(joint).squeeze(1)
        with torch.no_grad():
            mask_raw = cover.abs()
            mask = F.avg_pool1d(mask_raw.unsqueeze(1), kernel_size=11,
                                stride=1, padding=5).squeeze(1)
            mask = mask.clamp(min=0.005)
        modified = cover + perturbation_scale * delta * mask
        return modified.clamp(-1, 1), delta


class StegDecoder(nn.Module):
    def __init__(self, n_bits: int, hidden: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv1d(64, 96, 4, 2, 1), nn.GELU(),
            nn.Conv1d(96, hidden, 4, 2, 1), nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 60, 256), nn.GELU(),
            nn.Linear(256, n_bits),
        )

    def forward(self, audio):
        x = audio.unsqueeze(1)
        x = self.conv(x).flatten(1)
        return self.fc(x)


# ---------- bit-interleaved RS-FEC (copied from core/) ----------
class InterleavedRS:
    def __init__(self, n_data: int = 191, n_total: int = 255, depth: int = 8):
        assert (n_total - n_data) % 2 == 0
        self.n_data, self.n_total, self.depth = n_data, n_total, depth
        self.rs = reedsolo.RSCodec(n_total - n_data, nsize=n_total)

    def encode(self, data: bytes):
        block = self.n_data * self.depth
        pad = (-len(data)) % block
        if pad: data = data + bytes(np.random.bytes(pad))
        out = bytearray()
        for i in range(0, len(data), block):
            rows = [self.rs.encode(data[i+d*self.n_data : i+(d+1)*self.n_data])
                    for d in range(self.depth)]
            for col in range(self.n_total):
                for d in range(self.depth):
                    out.append(rows[d][col])
        return bytes(out), pad

    def decode(self, coded: bytes, pad: int):
        block = self.n_total * self.depth
        out = bytearray(); n_corr = 0; n_fail = 0
        for i in range(0, len(coded) - block + 1, block):
            seg = coded[i:i+block]
            rows = [bytearray() for _ in range(self.depth)]
            idx = 0
            for col in range(self.n_total):
                for d in range(self.depth):
                    rows[d].append(seg[idx]); idx += 1
            for r in rows:
                try:
                    dec = self.rs.decode(bytes(r))[0]; out += dec; n_corr += 1
                except reedsolo.ReedSolomonError:
                    out += bytes(r[:self.n_data]); n_fail += 1
        if pad and len(out) >= pad: out = out[:-pad]
        return bytes(out), n_corr, n_fail


# ---------- bit packing ----------
def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ---------- codec round-trips via ffmpeg (used at decode time, sender side
# normally just emits the WAV; round-trip is for self-test) ----------
def opus_round_trip(audio: np.ndarray, bitrate_kbps: int = 24,
                    application: str = "voip") -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.opus"; out = Path(tmp)/"out.wav"
        sf.write(inp, audio.astype(np.float32), SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(inp),
                        "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
                        "-application", application, str(opx)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(opx),
                        "-ar", str(SR), "-ac", "1", str(out)], check=True)
        a, _ = sf.read(out)
    a = a.astype(np.float32)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio): a = np.pad(a, (0, len(audio) - len(a)))
    elif len(a) > len(audio): a = a[:len(audio)]
    return a

def amrnb_round_trip(audio: np.ndarray, bitrate_kbps: float = 12.2) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.amr"; out = Path(tmp)/"out.wav"
        sf.write(inp, audio.astype(np.float32), SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(inp),
                        "-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1",
                        "-b:a", f"{bitrate_kbps}k", str(opx)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(opx),
                        "-ar", str(SR), "-ac", "1", str(out)], check=True)
        a, _ = sf.read(out)
    a = a.astype(np.float32)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio): a = np.pad(a, (0, len(audio) - len(a)))
    elif len(a) > len(audio): a = a[:len(audio)]
    return a


# ---------- repetition coding (used by AMR-NB pipeline) ----------
def rep_encode(bits: np.ndarray, n_rep: int) -> np.ndarray:
    return np.repeat(bits, n_rep)

def rep_decode(bits: np.ndarray, n_rep: int) -> np.ndarray:
    grouped = bits[: len(bits)//n_rep * n_rep].reshape(-1, n_rep)
    return (grouped.mean(axis=1) >= 0.5).astype(np.uint8)


# ---------- WAV helpers ----------
def numpy_to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def wav_bytes_to_numpy(data: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (WAV / WebM / Opus / MP4-AAC / etc.) to mono
    float32 at our SR. Browser MediaRecorder uploads are usually WebM/Opus."""
    # Fast path: native libsndfile read for WAV/FLAC/OGG.
    try:
        audio, sr = sf.read(io.BytesIO(data))
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr == SR:
            return audio
    except Exception:
        audio, sr = None, None

    # Fallback: hand the bytes to ffmpeg and let it figure out the format.
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp)/"in.bin"
        out = Path(tmp)/"out.wav"
        inp.write_bytes(data)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(inp),
                        "-ar", str(SR), "-ac", "1",
                        "-c:a", "pcm_s16le", str(out)], check=True)
        audio, _ = sf.read(out)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    return audio.astype(np.float32)


# ---------- pipelines ----------
@dataclass
class EncodeResult:
    wav_bytes: bytes
    n_data_bytes: int
    n_audio_seconds: float
    raw_bps: float
    eff_bps: float
    channel: str
    notes: str

@dataclass
class DecodeResult:
    text: str
    n_bytes_recovered: int
    raw_ber: float
    n_fec_blocks_corrected: int
    n_fec_blocks_failed: int
    channel: str


class OpusIIDPipeline:
    """IID neural codec, 4267 bps raw / 3196 bps reliable through Opus 24k VOIP."""
    name = "opus_iid"
    description = "IID neural codec — fastest rate, modem-tone carrier (Opus 24k VOIP target)"

    def __init__(self):
        ckpt = torch.load(CKPT_IID, map_location=DEVICE, weights_only=False)
        self.n_bits = ckpt["n_bits"]
        self.enc = Encoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.dec = Decoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.enc.load_state_dict(ckpt["encoder"])
        self.dec.load_state_dict(ckpt["decoder"])
        self.fec = InterleavedRS(n_data=191, n_total=255, depth=8)
        self.raw_bps = self.n_bits * 1000 / SYMBOL_MS  # 4266
        self.eff_bps = self.raw_bps * 191 / 255        # 3196

    @torch.no_grad()
    def encode_text(self, text: str) -> EncodeResult:
        data = text.encode("utf-8")
        coded, pad = self.fec.encode(data)
        tx_bits = bytes_to_bits(coded)
        pad_to_sym = (-len(tx_bits)) % self.n_bits
        if pad_to_sym:
            tx_bits = np.concatenate([tx_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
        n_sym = len(tx_bits) // self.n_bits
        symbols = tx_bits.reshape(n_sym, self.n_bits).astype(np.float32)
        sb = torch.from_numpy(symbols).to(DEVICE)
        audio = self.enc(sb).cpu().numpy().reshape(-1).astype(np.float32)
        wav = numpy_to_wav_bytes(audio)
        return EncodeResult(
            wav_bytes=wav,
            n_data_bytes=len(data),
            n_audio_seconds=len(audio) / SR,
            raw_bps=self.raw_bps,
            eff_bps=self.eff_bps,
            channel=self.name,
            notes=f"FEC RS(255,191)x8; {len(coded)} coded bytes encoded as {n_sym} symbols",
        )

    @torch.no_grad()
    def decode_audio(self, wav_bytes: bytes, expected_text_bytes: int | None = None) -> DecodeResult:
        audio = wav_bytes_to_numpy(wav_bytes)
        n_sym = len(audio) // SYMBOL_N
        if n_sym == 0:
            return DecodeResult("", 0, 1.0, 0, 0, self.name)
        chunks = audio[: n_sym * SYMBOL_N].reshape(n_sym, SYMBOL_N).astype(np.float32)
        sb = torch.from_numpy(chunks).to(DEVICE)
        # Process in batches so a long upload doesn't OOM
        rx_bits_chunks = []
        for i in range(0, n_sym, 256):
            logits = self.dec(sb[i:i+256])
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
        rx_bits = np.concatenate(rx_bits_chunks)
        rx_bytes = bits_to_bytes(rx_bits)
        # Try to FEC-decode; we don't know the original length so we'll trim aggressively
        decoded, n_corr, n_fail = self.fec.decode(rx_bytes, pad=0)
        # Strip null padding and best-effort trim to UTF-8
        text_guess = self._best_text(decoded, expected_text_bytes)
        # Raw BER undefined without knowing tx_bits, leave as nan-equivalent
        return DecodeResult(
            text=text_guess,
            n_bytes_recovered=len(decoded),
            raw_ber=float("nan"),
            n_fec_blocks_corrected=n_corr,
            n_fec_blocks_failed=n_fail,
            channel=self.name,
        )

    @staticmethod
    def _best_text(data: bytes, expected_len: int | None) -> str:
        if expected_len is not None:
            data = data[:expected_len]
        # Trim trailing NULs that came from FEC padding
        while data.endswith(b"\x00"):
            data = data[:-1]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


class StegoOpusPipeline:
    """Real-speech cover + neural stego, ~270 bps reliable through Opus 24k VOIP."""
    name = "stego_opus"
    description = "Real-speech cover with embedded data — sounds like a person talking with audible hiss"

    def __init__(self):
        ckpt = torch.load(CKPT_STEG, map_location=DEVICE, weights_only=False)
        self.n_bits = ckpt["n_bits"]
        self.pert_scale = ckpt.get("perturbation_scale", 0.5)
        self.enc = StegEncoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.dec = StegDecoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.enc.load_state_dict(ckpt["encoder"])
        self.dec.load_state_dict(ckpt["decoder"])
        # Strong FEC for the lower-rate channel
        self.fec = InterleavedRS(n_data=127, n_total=255, depth=8)
        self.raw_bps = self.n_bits * 1000 / SYMBOL_MS  # 1067
        self.eff_bps = self.raw_bps * 127 / 255        # ~531
        self._cover = self._load_cover()

    def _load_cover(self) -> np.ndarray:
        if not COVER_AUDIO.exists():
            raise FileNotFoundError(
                f"cover audio missing: {COVER_AUDIO} — bundle a clip into app/static/cover.wav"
            )
        cover, sr = sf.read(COVER_AUDIO)
        if cover.ndim > 1: cover = cover.mean(axis=1)
        if sr != SR:
            cover = self._resample(cover, sr, SR)
        return cover.astype(np.float32)

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp)/"in.wav"; out = Path(tmp)/"out.wav"
            sf.write(inp, audio, src_sr, format="WAV", subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(inp),
                            "-ar", str(dst_sr), "-ac", "1", str(out)], check=True)
            a, _ = sf.read(out)
        if a.ndim > 1: a = a.mean(axis=1)
        return a.astype(np.float32)

    def _cover_blocks(self, n_blocks: int) -> torch.Tensor:
        """Slice cover speech into n_blocks of 30ms; cycle if cover is shorter."""
        total_needed = n_blocks * SYMBOL_N
        if len(self._cover) >= total_needed:
            audio = self._cover[:total_needed]
        else:
            reps = int(np.ceil(total_needed / len(self._cover)))
            audio = np.tile(self._cover, reps)[:total_needed]
        return torch.from_numpy(audio.reshape(n_blocks, SYMBOL_N).astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def encode_text(self, text: str) -> EncodeResult:
        data = text.encode("utf-8")
        coded, pad = self.fec.encode(data)
        tx_bits = bytes_to_bits(coded)
        pad_to_sym = (-len(tx_bits)) % self.n_bits
        if pad_to_sym:
            tx_bits = np.concatenate([tx_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
        n_sym = len(tx_bits) // self.n_bits
        symbols = tx_bits.reshape(n_sym, self.n_bits).astype(np.float32)
        sb = torch.from_numpy(symbols).to(DEVICE)
        cover = self._cover_blocks(n_sym)
        modified, _ = self.enc(cover, sb, perturbation_scale=self.pert_scale)
        audio = modified.cpu().numpy().reshape(-1).astype(np.float32)
        wav = numpy_to_wav_bytes(audio)
        return EncodeResult(
            wav_bytes=wav,
            n_data_bytes=len(data),
            n_audio_seconds=len(audio) / SR,
            raw_bps=self.raw_bps,
            eff_bps=self.eff_bps,
            channel=self.name,
            notes=f"FEC RS(255,127)x8; cover speech with audible perturbation (~13 dB SNR)",
        )

    @torch.no_grad()
    def decode_audio(self, wav_bytes: bytes, expected_text_bytes: int | None = None) -> DecodeResult:
        audio = wav_bytes_to_numpy(wav_bytes)
        n_sym = len(audio) // SYMBOL_N
        if n_sym == 0:
            return DecodeResult("", 0, 1.0, 0, 0, self.name)
        chunks = audio[: n_sym * SYMBOL_N].reshape(n_sym, SYMBOL_N).astype(np.float32)
        sb = torch.from_numpy(chunks).to(DEVICE)
        rx_bits_chunks = []
        for i in range(0, n_sym, 256):
            logits = self.dec(sb[i:i+256])
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
        rx_bits = np.concatenate(rx_bits_chunks)
        rx_bytes = bits_to_bytes(rx_bits)
        decoded, n_corr, n_fail = self.fec.decode(rx_bytes, pad=0)
        text_guess = OpusIIDPipeline._best_text(decoded, expected_text_bytes)
        return DecodeResult(
            text=text_guess, n_bytes_recovered=len(decoded), raw_ber=float("nan"),
            n_fec_blocks_corrected=n_corr, n_fec_blocks_failed=n_fail, channel=self.name,
        )


class StegoAmrnbPipeline:
    """Real-speech cover + neural stego, ~76 bps reliable through AMR-NB cellular voice."""
    name = "stego_amrnb"
    description = "Cellular-voice (AMR-NB) variant — slowest, but only one that survives 2G/3G voice"

    def __init__(self):
        ckpt = torch.load(CKPT_AMR, map_location=DEVICE, weights_only=False)
        self.n_bits = ckpt["n_bits"]
        self.pert_scale = ckpt.get("perturbation_scale", 0.5)
        self.enc = StegEncoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.dec = StegDecoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.enc.load_state_dict(ckpt["encoder"])
        self.dec.load_state_dict(ckpt["decoder"])
        # No RS-FEC — we use 7x repetition coding inside the bit stream instead,
        # since at ~9% raw BER per symbol, byte BER is too high for any usable RS.
        self.n_rep = 7
        self.raw_bps = self.n_bits * 1000 / SYMBOL_MS  # 533
        self.eff_bps = self.raw_bps / self.n_rep       # ~76
        # cover audio loaded the same way as the Opus stego variant
        self._cover = StegoOpusPipeline._load_cover.__wrapped__(self) if False else None
        cover, sr = sf.read(COVER_AUDIO)
        if cover.ndim > 1: cover = cover.mean(axis=1)
        if sr != SR:
            cover = StegoOpusPipeline._resample(cover, sr, SR)
        self._cover = cover.astype(np.float32)

    def _cover_blocks(self, n_blocks: int) -> torch.Tensor:
        total = n_blocks * SYMBOL_N
        if len(self._cover) >= total:
            audio = self._cover[:total]
        else:
            audio = np.tile(self._cover, int(np.ceil(total / len(self._cover))))[:total]
        return torch.from_numpy(audio.reshape(n_blocks, SYMBOL_N).astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def encode_text(self, text: str) -> EncodeResult:
        data = text.encode("utf-8")
        data_bits = bytes_to_bits(data)
        rep_bits = rep_encode(data_bits, self.n_rep)
        pad_to_sym = (-len(rep_bits)) % self.n_bits
        if pad_to_sym:
            rep_bits = np.concatenate([rep_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
        n_sym = len(rep_bits) // self.n_bits
        symbols = rep_bits.reshape(n_sym, self.n_bits).astype(np.float32)
        sb = torch.from_numpy(symbols).to(DEVICE)
        cover = self._cover_blocks(n_sym)
        modified, _ = self.enc(cover, sb, perturbation_scale=self.pert_scale)
        audio = modified.cpu().numpy().reshape(-1).astype(np.float32)
        wav = numpy_to_wav_bytes(audio)
        return EncodeResult(
            wav_bytes=wav,
            n_data_bytes=len(data),
            n_audio_seconds=len(audio) / SR,
            raw_bps=self.raw_bps,
            eff_bps=self.eff_bps,
            channel=self.name,
            notes=f"7x repetition coding (no RS); cover speech with ~7 dB SNR perturbation",
        )

    @torch.no_grad()
    def decode_audio(self, wav_bytes: bytes, expected_text_bytes: int | None = None) -> DecodeResult:
        audio = wav_bytes_to_numpy(wav_bytes)
        n_sym = len(audio) // SYMBOL_N
        if n_sym == 0:
            return DecodeResult("", 0, 1.0, 0, 0, self.name)
        chunks = audio[: n_sym * SYMBOL_N].reshape(n_sym, SYMBOL_N).astype(np.float32)
        sb = torch.from_numpy(chunks).to(DEVICE)
        rx_bits_chunks = []
        for i in range(0, n_sym, 256):
            logits = self.dec(sb[i:i+256])
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
        rx_rep_bits = np.concatenate(rx_bits_chunks)
        # Majority-vote decode the repetition
        usable = (len(rx_rep_bits) // self.n_rep) * self.n_rep
        data_bits = rep_decode(rx_rep_bits[:usable], self.n_rep)
        decoded = bits_to_bytes(data_bits)
        text_guess = OpusIIDPipeline._best_text(decoded, expected_text_bytes)
        return DecodeResult(
            text=text_guess, n_bytes_recovered=len(decoded), raw_ber=float("nan"),
            n_fec_blocks_corrected=0, n_fec_blocks_failed=0, channel=self.name,
        )


# ---------- registry ----------
PIPELINES: dict[str, type] = {
    OpusIIDPipeline.name:     OpusIIDPipeline,
    StegoOpusPipeline.name:   StegoOpusPipeline,
    StegoAmrnbPipeline.name:  StegoAmrnbPipeline,
}

# Singletons populated lazily by get()
_INSTANCES: dict = {}

def get_pipeline(name: str):
    """Lazy-load a pipeline by name. Caches the instance for subsequent calls."""
    if name not in PIPELINES:
        raise KeyError(f"unknown pipeline {name!r}; available: {list(PIPELINES)}")
    if name not in _INSTANCES:
        _INSTANCES[name] = PIPELINES[name]()
    return _INSTANCES[name]


def list_pipelines() -> list[dict]:
    """Metadata for every pipeline; safe to call without instantiating models."""
    return [
        {
            "name": cls.name,
            "description": cls.description,
        }
        for cls in PIPELINES.values()
    ]
