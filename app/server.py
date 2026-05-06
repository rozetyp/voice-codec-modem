"""
FastAPI server for the voice-channel-poc demo.

  GET  /              -> HTML page with three radio buttons + textarea + audio player
  GET  /healthz       -> {"ok": true}  (Railway healthcheck)
  GET  /channels      -> list of available channels
  POST /encode        -> {channel, text}            -> WAV file (audio/wav)
  POST /decode        -> multipart {channel, file}  -> {decoded_text, ...}

Notes:
  * Pipelines lazy-load on first use (saves ~3-5 sec on a cold start).
  * The /encode response also returns the original text length in headers
    so /decode can trim recovered bytes to the exact length.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import pipelines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-channel-poc")

app = FastAPI(title="voice-channel-poc",
              description="Data-over-voice channel demo. Encode short text into "
                          "audio, play it through a real voice channel, then upload "
                          "the receiver-side recording to decode it back.",
              version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/channels")
def channels():
    return JSONResponse({"channels": pipelines.list_pipelines()})


@app.post("/encode")
def encode(channel: str = Form(...), text: str = Form(...)):
    """Encode `text` to audio via the selected channel. Returns audio/wav."""
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if channel not in pipelines.PIPELINES:
        raise HTTPException(status_code=400, detail=f"unknown channel '{channel}'; "
                                                    f"available: {list(pipelines.PIPELINES)}")
    log.info(f"/encode channel={channel} text_len={len(text)}")
    try:
        pipe = pipelines.get_pipeline(channel)
        result = pipe.encode_text(text)
    except Exception as exc:  # narrow only after we see what real failures look like
        log.exception("encode failed")
        raise HTTPException(status_code=500, detail=str(exc))
    headers = {
        "X-Channel": channel,
        "X-Text-Bytes": str(result.n_data_bytes),
        "X-Audio-Seconds": f"{result.n_audio_seconds:.3f}",
        "X-Raw-Bps": f"{result.raw_bps:.0f}",
        "X-Eff-Bps": f"{result.eff_bps:.0f}",
        "X-Notes": result.notes,
        "Content-Disposition": f'attachment; filename="encoded-{channel}.wav"',
    }
    return Response(content=result.wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/decode")
async def decode(channel: str = Form(...),
                 expected_text_bytes: int | None = Form(default=None),
                 file: UploadFile = File(...)):
    """Decode an audio upload. `expected_text_bytes` is optional but helps trim
    trailing FEC padding cleanly; the /encode response includes it in X-Text-Bytes."""
    if channel not in pipelines.PIPELINES:
        raise HTTPException(status_code=400, detail=f"unknown channel '{channel}'")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    log.info(f"/decode channel={channel} bytes={len(raw)} expected_text_bytes={expected_text_bytes}")
    try:
        pipe = pipelines.get_pipeline(channel)
        result = pipe.decode_audio(raw, expected_text_bytes=expected_text_bytes)
    except Exception as exc:
        log.exception("decode failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse({
        "channel": result.channel,
        "decoded_text": result.text,
        "n_bytes_recovered": result.n_bytes_recovered,
        "n_fec_blocks_corrected": result.n_fec_blocks_corrected,
        "n_fec_blocks_failed": result.n_fec_blocks_failed,
    })


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>voice-channel-poc</h1><p>Static UI not bundled. "
        "POST to /encode and /decode directly.</p>"
    )
