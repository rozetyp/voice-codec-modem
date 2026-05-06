"""
Voice-within-voice demo.

A second voice (Codec2 700C-encoded) is hidden inside a real-speech cover via
the trained stego_opus model, sent through Opus 24k VOIP (Zoom-class), and
recovered on the other side.

The cover party hears: regular speech with a faint hiss (the stego perturbation).
The decoder party hears: the cover speech AND, after running the decoder,
                          the second voice synthesized from Codec2.

Pipeline:
  hidden.wav  -- c2enc 700C (700 bps) -->  hidden.c2 (~88 bytes/sec)
                 -- 4-byte length prefix --  payload bytes
                 -- RS(255, 127) × depth-8 interleaving --  coded bytes
                 -- bit pack, 32 bits / 30 ms symbol --  symbols
                 -- StegEncoder over real-speech cover --  audio_stego
                 -- Opus 24k VOIP round-trip --  audio_received
                 -- StegDecoder --  symbols
                 -- bit unpack, RS decode, strip prefix --  hidden.c2 (recovered)
                 -- c2dec 700C --  hidden_recovered.wav

End-to-end this is ~531 bps reliable. Codec2 700C runs at 700 bps but our test
clip is short (3.9 s of voice in a 30 s cover) so the time-averaged demand on
the channel is well below capacity.
"""
from __future__ import annotations
import argparse, io, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipelines import (
    StegEncoder, StegDecoder, InterleavedRS,
    bytes_to_bits, bits_to_bytes, SR, SYMBOL_N, SYMBOL_MS, DEVICE,
    CKPT_STEG, COVER_AUDIO,
)

OPUS_KBPS = 24
OPUS_APP = "voip"


# ---------- small opus round-trip helper ----------
def opus_rt(audio: np.ndarray, sr: int = SR,
            kbps: int = OPUS_KBPS, app: str = OPUS_APP) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.opus"; out = Path(tmp)/"out.wav"
        sf.write(inp, audio.astype(np.float32), sr)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a",f"{kbps}k","-application",app,
                        str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(sr),"-ac","1",str(out)], check=True)
        a, _ = sf.read(out)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio): a = np.pad(a, (0, len(audio)-len(a)))
    elif len(a) > len(audio): a = a[:len(audio)]
    return a.astype(np.float32)


# ---------- transmit / receive arbitrary bytes via stego ----------
class StegoBytesTransport:
    """Just the byte-transport guts of StegoOpusPipeline, exposed.
    Pure ML — no text-encoding. Used for the v-w-v demo where we transmit
    Codec2 binary bytes, not text."""

    def __init__(self):
        ckpt = torch.load(CKPT_STEG, map_location=DEVICE, weights_only=False)
        self.n_bits = ckpt["n_bits"]
        self.pert_scale = ckpt.get("perturbation_scale", 0.5)
        self.enc = StegEncoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.dec = StegDecoder(n_bits=self.n_bits).to(DEVICE).eval()
        self.enc.load_state_dict(ckpt["encoder"])
        self.dec.load_state_dict(ckpt["decoder"])
        self.fec = InterleavedRS(n_data=127, n_total=255, depth=8)
        # Rate accounting
        self.raw_bps = self.n_bits * 1000 / SYMBOL_MS                         # 1067
        self.eff_bps = self.raw_bps * self.fec.n_data / self.fec.n_total      # ~531

        # cover
        cover, sr = sf.read(COVER_AUDIO)
        if cover.ndim > 1: cover = cover.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            cover = resample_poly(cover, SR, sr)
        self._cover = cover.astype(np.float32)

    def _cover_blocks(self, n_blocks: int) -> torch.Tensor:
        total = n_blocks * SYMBOL_N
        if len(self._cover) >= total:
            audio = self._cover[:total]
        else:
            reps = int(np.ceil(total / len(self._cover)))
            audio = np.tile(self._cover, reps)[:total]
        return torch.from_numpy(audio.reshape(n_blocks, SYMBOL_N).astype(np.float32)).to(DEVICE)

    def capacity_bytes_for(self, audio_seconds: float) -> int:
        return int(self.eff_bps * audio_seconds / 8)

    @torch.no_grad()
    def send(self, payload: bytes) -> tuple[np.ndarray, int]:
        """Encode payload + 4-byte length prefix → audio. Returns (audio, pad)."""
        framed = struct.pack("<I", len(payload)) + payload
        coded, pad = self.fec.encode(framed)
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
        return audio, pad

    @torch.no_grad()
    def recv(self, audio: np.ndarray, pad: int) -> tuple[bytes, dict]:
        n_sym = len(audio) // SYMBOL_N
        if n_sym == 0:
            return b"", {"error": "audio too short"}
        chunks = audio[: n_sym * SYMBOL_N].reshape(n_sym, SYMBOL_N).astype(np.float32)
        sb = torch.from_numpy(chunks).to(DEVICE)
        rx_bits_chunks = []
        for i in range(0, n_sym, 256):
            logits = self.dec(sb[i:i+256])
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
        rx_bits = np.concatenate(rx_bits_chunks)
        rx_bytes = bits_to_bytes(rx_bits)
        decoded, n_corr, n_fail = self.fec.decode(rx_bytes, pad=pad)
        # Strip 4-byte length prefix and trim
        if len(decoded) >= 4:
            length = struct.unpack("<I", decoded[:4])[0]
            if 0 <= length <= len(decoded) - 4:
                payload = decoded[4:4+length]
            else:
                payload = decoded[4:]   # length corrupted; return as-is
        else:
            payload = b""
        return payload, {"fec_corrected": n_corr, "fec_failed": n_fail,
                         "length_field": length if len(decoded) >= 4 else -1}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="/tmp/hidden.raw",
                    help="raw signed-16 8 kHz PCM of the hidden voice")
    ap.add_argument("--out-dir", default="/tmp/voice_within_voice")
    ap.add_argument("--c2-mode", default="700C")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # Step 1: Codec2-encode the hidden voice
    hidden_c2 = out_dir/"hidden.c2"
    print(f"[1] Codec2 {args.c2_mode} encoding {args.hidden} -> {hidden_c2}")
    subprocess.run(["c2enc", args.c2_mode, args.hidden, str(hidden_c2)], check=True)
    payload = hidden_c2.read_bytes()
    print(f"    payload: {len(payload)} bytes")

    # Step 2: Build transport, check fit
    print("\n[2] Loading StegoBytesTransport…")
    tx = StegoBytesTransport()
    print(f"    eff_bps={tx.eff_bps:.0f}  raw_bps={tx.raw_bps:.0f}  cover={len(tx._cover)/SR:.1f}s")
    cover_capacity_bytes = tx.capacity_bytes_for(len(tx._cover) / SR)
    print(f"    cover capacity: {cover_capacity_bytes} bytes ({len(payload)/cover_capacity_bytes*100:.0f}% utilization)")
    assert len(payload) + 4 < cover_capacity_bytes, \
        f"payload {len(payload)+4}B exceeds cover capacity {cover_capacity_bytes}B"

    # Step 3: Encode through stego
    print("\n[3] StegoEncoder over real-speech cover…")
    audio_tx, pad = tx.send(payload)
    secs = len(audio_tx) / SR
    print(f"    audio: {secs:.1f}s  pad={pad}B")
    sf.write(out_dir/"tx.wav", audio_tx, SR)

    # Step 4: Opus 24k VOIP round-trip
    print(f"\n[4] Opus {OPUS_KBPS}k {OPUS_APP} round-trip…")
    audio_rx = opus_rt(audio_tx)
    sf.write(out_dir/"rx.wav", audio_rx, SR)

    # Step 5: Decode bytes
    print("\n[5] StegoDecoder…")
    recovered, info = tx.recv(audio_rx, pad)
    print(f"    recovered: {len(recovered)} bytes (target {len(payload)})  "
          f"fec_corrected={info.get('fec_corrected')}  fec_failed={info.get('fec_failed')}  "
          f"length_field={info.get('length_field')}")

    if len(recovered) != len(payload):
        print(f"    ⚠ size mismatch (got {len(recovered)}, expected {len(payload)})")
    n_match_bytes = sum(a == b for a, b in zip(recovered, payload))
    print(f"    bytes matching: {n_match_bytes}/{min(len(recovered), len(payload))}")

    # Save recovered bytes and run codec2 decoder
    recovered_c2 = out_dir/"hidden_recovered.c2"
    recovered_c2.write_bytes(recovered)
    recovered_raw = out_dir/"hidden_recovered.raw"
    print(f"\n[6] Codec2 {args.c2_mode} decode -> {recovered_raw}")
    rc = subprocess.run(["c2dec", args.c2_mode, str(recovered_c2), str(recovered_raw)],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"    c2dec failed: {rc.stderr}")
        return
    print(f"    c2dec stdout: {rc.stdout.strip()}")

    # Convert raw to WAV for easy listening
    recovered_wav = out_dir/"hidden_recovered.wav"
    subprocess.run(["ffmpeg","-y","-loglevel","error",
                    "-f","s16le","-ar","8000","-ac","1",
                    "-i", str(recovered_raw), str(recovered_wav)], check=True)
    print(f"\n=== outputs in {out_dir} ===")
    print(f"  tx.wav                 : encoder output (cover + hidden voice, 16 kHz)")
    print(f"  rx.wav                 : Opus round-trip of tx.wav")
    print(f"  hidden_recovered.wav   : the hidden voice recovered from rx.wav")


if __name__ == "__main__":
    main()
