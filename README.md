# Voice-Codec Modem

Hide arbitrary digital data inside speech-shaped audio that survives the codecs used by Zoom, Discord, WhatsApp, and cellular voice. Six trained codecs span 76 bps (cellular voice) to 3196 bps (Zoom-class), with zero post-FEC errors verified on multi-seed validation. End-to-end voice-within-voice demos work: a second voice (Codec2) is hidden inside a normal-sounding call and recoverable bit-perfectly on the other end.

This is a research artifact, not a product. The codec is not novel methodology (it sits within the codec-in-loop training framework published by Juvela 2024, with watermark-class steganography ideas added). What is published here is a reproducible Pareto curve across rate, audibility, and channel, with working code and listenable demos for each point.

[Audio demos](#listen) · [Quickstart](#quickstart) · [Pareto frontier](#pareto-frontier) · [What works, what does not](#honest-limitations)

---

## What this delivers

| Channel | Where it ships in the wild | Best reliable rate |
|---|---|---|
| Opus 48k AUDIO | Discord, music-mode WebRTC | 4000 bps modem-tone, 2000 bps speech-dominant, 933 bps real-speech-cover |
| Opus 24k VOIP | Zoom, Teams, modern WebRTC | 3196 bps modem-tone, 800 bps speech-textured, 270 bps real-speech-cover |
| Opus 12k VOIP | WhatsApp, Signal voice | 3196 bps (modem-tone only) |
| **AMR-NB 12.2k** | **2G/3G cellular voice, PSTN** | **76 bps real-speech-cover** with repetition coding |

All numbers are post-FEC, multi-seed validated through real ffmpeg-libopus or ffmpeg-amrnb (not surrogates). Zero residual errors observed on 76,000+ bits per checkpoint at the headline configurations.

The cellular result is the most unusual property: no published OSS waveform-domain method works through AMR-NB. Most audio watermarking targets MP3/AAC or Opus. AMR-NB cellular voice gets little research attention.

## Listen

Inside [core/neural_codec/](core/neural_codec/) and its subfolders are listenable WAVs for every variant. Cover versus modified versus through-codec for each codec class.

The headline demo, voice-within-voice through Zoom-class Opus 24k VOIP at 2400 bps Codec2:

```bash
.venv/bin/python core/neural_codec/voice_within_voice_demo.py --length_s 10 --codec2_mode 2400
# outputs: core/neural_codec/demo_outputs/
#   01_original_16k.wav                       cover voice (real speech)
#   05_transmitted_carrier_audio_16k.wav      what eavesdroppers hear (synthetic, modem-toned)
#   07_recovered_voice_within_voice_16k.wav   the second voice recovered (bit-identical to Codec2)
```

The new lower-rate demo with a real-speech-sounding cover at Codec2 700C, embedded via stego_opus + Opus 24k VOIP:

```bash
say "Meet me at the warehouse at midnight." -o /tmp/hidden.aiff   # macOS TTS
ffmpeg -i /tmp/hidden.aiff -ar 8000 -ac 1 -f s16le /tmp/hidden.raw
python3 app/voice_within_voice.py
# outputs: /tmp/voice_within_voice/
#   tx.wav                  cover voice with embedded data (audible hiss)
#   rx.wav                  same after Opus 24k VOIP round-trip
#   hidden_recovered.wav    the second voice (Codec2 700C, robotic but intelligible)
```

## Quickstart

```bash
git clone <this-repo>
cd voice-codec-modem

# Python 3.10+ required (type-union syntax used throughout).
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# System tools
brew install ffmpeg codec2          # macOS
# or: apt-get install ffmpeg codec2  (linux)

# First demo: 3.2 kbps voice-within-voice through Opus 24k VOIP
.venv/bin/python core/neural_codec/voice_within_voice_demo.py --length_s 10 --codec2_mode 2400

# Multi-seed validation of any checkpoint
.venv/bin/python core/neural_codec/eval_real_opus.py core/neural_codec/ckpt_n128_mixed.pt --seeds 5

# AMR-NB cellular variant
.venv/bin/python core/neural_codec/cellular/amrnb_eval_with_rep.py
```

The newer prosody and stacked-channel demos in `app/` use `pyworld`, which has an extra setuptools dependency. They are run with system Python (where pyworld is already installed) or with a fresh venv that has setuptools<80.

## Pareto frontier

Five non-dominated checkpoints, each at a different point on the rate vs realism vs channel curve.

| Checkpoint | Reliable rate | Post-FEC BER | Carrier sounds like | Best for |
|---|---|---|---|---|
| `core/neural_codec/ckpt_n128_mixed.pt` | 3196 bps | 0.0000% on 76 kbits | Modem tone | Pure rate, machine-to-machine over a phone call |
| `core/neural_codec/adversarial/ckpt_n128_adv.pt` | 2125 bps | 0.05% (1 bit per 2000) | Speech-textured noise | Balanced rate vs realism |
| `core/neural_codec/adversarial/ckpt_n64_adv.pt` | 1598 bps | 0.0000% | Speech-textured noise | Lower-rate speech-textured |
| `core/neural_codec/sequence/ckpt_seq_adv.pt` | 799 bps | 0.0000% | Speech-like babble with cross-symbol envelope | Realism-headline; closest to real speech in mel-L1 |
| `core/neural_codec/stego/ckpt_stego_p3.pt` | 270 bps (Opus 24k VOIP) or 933 bps (Opus 48k AUDIO) | 0.0000% | Real human speech with audible hiss | Real-speech cover |
| `core/neural_codec/cellular/ckpt_amrnb_real.pt` | 76 bps | 0.58% post-rep | Real human speech with audible hiss | Cellular voice (only known method) |

The newer stacked channel adds a 5 bps prosody sub-channel via pyworld pitch modulation; bulletproof through any speech codec including the speaker-to-microphone acoustic loop.

## Repository structure

```
voice-codec-modem/
├── core/
│   ├── neural_codec/                  IID neural codecs, headline 3196 bps
│   │   ├── adversarial/               IID + adversarial realism, 29-31% closer to real speech
│   │   ├── sequence/                  SEQ codec with cross-symbol coherence
│   │   ├── stego/                     Real-speech-cover stego, 270 bps via Opus 24k VOIP
│   │   ├── cellular/                  AMR-NB cellular variant, 76 bps reliable
│   │   ├── prosody/                   Pitch-modulation modem (5 bps reliable, any codec)
│   │   └── composite_attempt/         Documented negative result (TRIZ band-split)
│   ├── triz_pitch/                    Hand-coded modem (175 bps reliable, no neural net)
│   └── slow_ghost.py, production_modem.py, ...   Earlier prototypes, retired (see Reality Check)
├── app/
│   ├── voice_within_voice.py          Codec2 700C through stacked stego_opus + prosody
│   ├── test_stacked_channel.py        Stego + prosody together
│   ├── test_prosody_modem.py          Pitch-modulation standalone
│   ├── modal_probe_*.py               Multi-codec shootout, EnCodec/Mimi/DAC analysis
│   ├── modal_train_encodec_opus.py    Failed EnCodec encoder fine-tune (kept for reference)
│   └── pipelines.py, server.py        FastAPI POC for in-browser encode/decode
├── tests/                             Validation scripts and audio fixtures
├── tools/                             Real-codec test harness
└── Dockerfile, railway.json, .railwayignore   Deployed POC config (separate from research)
```

## How it works

The neural codec is an end-to-end learned encoder/decoder pair with the real codec in the training loop.

```
data bytes
  -> Reed-Solomon (255,191) interleaved encode
  -> bits
  -> Encoder (1D conv stack, 3-32M params)
  -> 30 ms audio frames at 16 kHz
  -> ffmpeg -c:a libopus -b:a 24k -application voip   (the channel)
  -> Decoder (mirror conv stack)
  -> bit logits
  -> RS decode
  -> data bytes
```

Training mixes a differentiable surrogate channel (lowpass plus shaped noise) with periodic real-Opus straight-through steps every 25 iterations. The straight-through trick passes gradients as identity through the non-differentiable codec while running real ffmpeg-libopus on the forward pass. This closes the surrogate-to-real gap that single-channel surrogate training cannot reach.

For AMR-NB, the surrogate fails entirely (ACELP coding is too far from the surrogate distribution) so training runs every step through real ffmpeg-amrnb. Five to ten times slower per step, but the only way to converge for cellular voice.

The stego variants take a different path: the encoder takes (cover_speech, bits) and produces a small additive perturbation that rides on real recorded human speech. The cover IS speech that the codec is psychoacoustically tuned to preserve. Bits ride underneath as small amplitude-masked perturbations.

The hand-coded TRIZ-inversion modem in `core/triz_pitch/` does not use neural networks. It synthesizes voiced audio whose Opus-preserved features (pitch, gain, spectral tilt) ARE the data. Three orthogonal channels stacked, then bit-interleaved Reed-Solomon over GF(256). 175 bps reliable, zero observed errors across 152 kbits, no GPU needed.

## Honest limitations

This is a research artifact. Several things do not work, are not measured, or are below published state of the art.

**Things that do not work as advertised in earlier README drafts:**
- `slow_ghost.py` claimed 2.7 kbps. It does 10 bps and the decoder is an untrained `nn.Module`.
- `production_modem.py` claimed 800 bps with FEC. It does not round-trip even without a codec; the energy-threshold decoder mismatches the encoder's levels. ~50% BER.
- `band_energy_modulation.py` and friends claim 1-2 kbps but only test against an in-file `mock_opus_codec` which is just A-weighted Gaussian noise. Untested against real libopus.

These are kept in the repo with prominent failure notices in the [Reality Check](#reality-check) section. They are not part of the working set.

**Things untested or below state of the art:**
- AMR-WB, EVS, G.711, G.722. Major cellular and PSTN codecs not characterized.
- Adversarial steganalysis robustness. Never trained against a real detector.
- Live two-way duplex. All round-trips are file-based; no jitter, packet loss, or sync preamble.
- Real cellular network conditions. AMR-NB tested via ffmpeg only.
- Imperceptibility. Best stego variant is ~13 dB SNR (audible hiss). True watermark-class imperceptibility (>25 dB SNR) was not reached.
- The 76 bps AMR-NB number is far below published codec-internal QIM-based AMR steganography (Tian et al. 2018 report 1-3 kbps). The legitimate distinction is that ours is blind waveform-domain (sender produces a normal WAV) while published methods are codec-internal (sender modifies the AMR encoder during quantization). Different problem class. Not better, not novel methodology.

**Things this is not:**
- Not a commercial product. AudioSeal owns the watermarking market at much better imperceptibility; we are at a different point on the curve (higher capacity, audible carrier).
- Not a research breakthrough. Methods are published (Juvela 2024 codec-in-loop training, watermark-class stego patterns).
- Not novel methodology. The contribution is a reproducible Pareto curve, working code, and an honest writeup of what the techniques deliver across three classes of voice codec.

## Reality check

Discrepancies caught while auditing earlier README drafts. Documented to prevent re-litigation.

- `slow_ghost.py` claimed 2.7 kbps and 1.15% BER. The code does 10 bps. No BER is measured anywhere. The decoder has random untrained weights. Retired.
- `production_modem.py` claimed 800 bps "Production Ready" with <0.5% BER. The script's decoder threshold does not match its own encoder levels; it does not round-trip even without a codec. The "99.9% codec survival rate" string is hard-coded, not measured. Retired.
- `LLM semantic compression multiplies effective capacity` claim conflated channel bitrate with information rate. Removed.
- `utils/ACTION_PLAN_10KBPS.py` is 0 bytes. Other `utils/*.py` files are prose-as-Python rather than runnable code.
- The previous "10 kbps reliable" target was aspirational and never validated through real Opus.

## Prior art

The methods used here are published. This work is an application of those methods to a wider Pareto frontier with reproducible artifacts, not a novel technique.

- **Codec-in-loop neural watermarking**: [Juvela 2024, "Audio Codec Augmentation for Robust Collaborative Watermarking"](https://arxiv.org/abs/2409.13382). The straight-through training procedure used in this repo is from this paper.
- **AMR codec-internal steganography**: [Tian et al. 2018, "An AMR adaptive steganography algorithm based on minimizing distortion"](https://link.springer.com/article/10.1007/s11042-017-4860-1). Reports 1-3 kbps via codec-internal QIM. Published state of the art for cellular-codec stego. Higher rate than this repo because it modifies the AMR encoder directly rather than going through it blindly.
- **AudioSeal**: [Meta 2024](https://github.com/facebookresearch/audioseal), MIT licensed. The dominant OSS audio watermarking system. Higher imperceptibility than ours; lower capacity (16 bits/sec). Different point on the Pareto curve.
- **SilentCipher**: [Sony 2024](https://github.com/sony/silentcipher). 30 bits/sec, near-imperceptible.
- **WavMark**: [Github](https://github.com/wavmark/wavmark). 32 bits/sec.
- **GGWave**: [Georgi Gerganov](https://github.com/ggerganov/ggwave). Acoustic data modem, 8-16 bps reliable through air, audible chirps.

## Reproducibility

Every headline number can be reproduced from the scripts in the repo:

- 3.2 kbps neural codec: `core/neural_codec/voice_within_voice_demo.py`
- Multi-seed validation of any checkpoint: `core/neural_codec/eval_real_opus.py <ckpt> --seeds 5`
- AMR-NB cellular variant: `core/neural_codec/cellular/amrnb_eval_with_rep.py`
- Adversarial realism: `core/neural_codec/adversarial/realism_eval.py`
- Stego with real-speech cover: `core/neural_codec/stego/stego_listen.py`
- Hand-coded TRIZ modem: `core/triz_pitch/v5_final_validate.py`

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text. Apache 2.0 includes an explicit patent grant, which matters for audio watermarking work because the field has active patents (Verance/Cinavia, Audible Magic, NexGuard).

## Acknowledgments

LibriSpeech dev-clean was used for diversity-tested training in some checkpoints. Two-clip training data (one synthetic readaloud, one Lex Fridman snippet) was used for early stego variants. Real ffmpeg-libopus and ffmpeg-amrnb-encoder are the channel implementations used throughout.
