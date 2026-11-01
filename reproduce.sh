#!/usr/bin/env bash
# Re-run every result in the README's Results table, through real ffmpeg codecs, and check it.
# Usage: ./reproduce.sh            (PYTHON=python ./reproduce.sh outside the .venv)
# Rows whose dependency is missing (AMR-NB ffmpeg, opuslib, codec2, pyworld) are skipped, not failed.
# Exit status is the number of rows whose check failed.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ "$(uname)" = Darwin ] && export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"
NC=core/neural_codec
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
SEQ_URL=https://github.com/rozetyp/voice-codec-modem/releases/download/v0.2.0/ckpt_seq_adv.pt
SEQ_SHA=ea718d17e69403f42dfcea5e84843c3f1db4319ed5dff1f838babd6371891b04
FAILS=0; RESULTS=()

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
# run "<row label>" "<check>" cmd...
#   check = a number: max post-FEC payload BER in percent, read from the "(errors/bits bits)" total
#   check = text:     a regex that must appear in the output
#   check = empty:    report only
run() {
  local label="$1" check="$2"; shift 2
  printf '\n==== %s\n' "$label"
  "$@" 2>&1 | grep -v -e FutureWarning -e 'torch.load(' | tee "$TMP/out"
  local verdict=PASS
  if [ -z "$check" ]; then verdict=REPORT
  elif [[ "$check" =~ ^[0-9.]+$ ]]; then
    local eb; eb=$(grep -oE '\([0-9]+/[0-9]+ bits\)' "$TMP/out" | tail -1 | tr -d '( bits)')
    if [ -z "$eb" ] || ! awk -v e="${eb%/*}" -v b="${eb#*/}" -v m="$check" 'BEGIN { exit !(100 * e / b <= m) }'; then verdict=FAIL; fi
    label="$label  [${eb:-no total}, limit ${check}%]"
  elif ! grep -qE "$check" "$TMP/out"; then verdict=FAIL; fi
  RESULTS+=("$verdict  $label"); [ "$verdict" = FAIL ] && FAILS=$((FAILS + 1)); return 0
}
skip() { printf '\n==== %s\nSKIPPED: %s\n' "$1" "$2"; RESULTS+=("SKIP  $1 ($2)"); }

run "3196 bps | modem tone | Opus 24k VOIP" 0 \
  "$PY" $NC/neural_with_fec.py $NC/ckpt_n128_mixed.pt --seeds 5
run "2125 bps | speech-textured noise | Opus 24k VOIP" 0.2 \
  "$PY" $NC/neural_with_fec.py $NC/adversarial/ckpt_n128_adv.pt --seeds 5 --rs_data 127
run "1330 bps | speech-textured noise | Opus 24k VOIP" 0 \
  "$PY" $NC/neural_with_fec.py $NC/adversarial/ckpt_n64_adv.pt --seeds 5 --rs_data 159

if [ ! -f $NC/sequence/ckpt_seq_adv.pt ]; then curl -fsSL -o $NC/sequence/ckpt_seq_adv.pt "$SEQ_URL" || true; fi
if [ -f $NC/sequence/ckpt_seq_adv.pt ] && [ "$(sha256 $NC/sequence/ckpt_seq_adv.pt)" = "$SEQ_SHA" ]; then
  run "799 bps | speech-like babble | Opus 24k VOIP" 0.2 \
    "$PY" $NC/sequence/seq_with_fec.py $NC/sequence/ckpt_seq_adv.pt --seeds 5
else skip "799 bps | speech-like babble | Opus 24k VOIP" "ckpt_seq_adv.pt missing or checksum mismatch"; fi

run "531 bps | real speech + hiss | Opus 24k VOIP" 0.5 \
  "$PY" $NC/stego/stego_with_fec.py --application voip --kbps 24 --rs_data 127
run "799 bps | real speech + hiss | Opus 48k AUDIO" 0 \
  "$PY" $NC/stego/stego_with_fec.py --application audio --kbps 48 --rs_data 191

if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libopencore_amrnb; then
  run "AMR-NB 12.2k | real speech + hiss | repetition coding (report only)" '' \
    "$PY" $NC/cellular/amrnb_eval_with_rep.py
else skip "AMR-NB 12.2k | real speech + hiss" "this ffmpeg has no libopencore_amrnb (Homebrew's doesn't; Debian/Ubuntu's does)"; fi

if command -v c2enc >/dev/null; then
  run "Voice within voice | Codec2 2400 inside the 3196 bps carrier" 'EXACTLY recovered' \
    "$PY" $NC/voice_within_voice_demo.py --length_s 10 --codec2_mode 2400 --out_dir "$TMP/vwv"
else skip "Voice within voice" "codec2 (c2enc) not installed"; fi

if "$PY" -c "import opuslib" 2>/dev/null; then
  run "175 bps | hand-coded voiced tone, no neural net | Opus 24k VOIP" 'WINNER .* 0\.0000% ' \
    "$PY" core/triz_pitch/v5_final_validate.py
else skip "175 bps | hand-coded voiced tone" "opuslib / libopus not importable (see requirements.txt)"; fi

if "$PY" -c "import pyworld" 2>/dev/null; then
  run "~4 bps raw | pitch wobble on real speech, no FEC (report only)" '' \
    "$PY" experiments/test_prosody_modem.py
else skip "~4 bps raw | pitch wobble" "pyworld not installed"; fi

printf '\n==== Summary\n'; printf '%s\n' "${RESULTS[@]}"
exit "$FAILS"
