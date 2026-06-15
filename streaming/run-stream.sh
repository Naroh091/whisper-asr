#!/bin/bash
# Wrapper para supervisord: servicio de transcripción+diarización en streaming (:18006).
# Corre en diart-venv (CPU). Reutiliza el Whisper de GPU del backend batch (:18005)
# para el texto y diart (pyannote 3.1) para la diarización online.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ASR_STREAM_VENV:-/workspace/diart-venv}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export ASR_RT_HOST="${ASR_RT_HOST:-127.0.0.1}"
export ASR_RT_PORT="${ASR_RT_PORT:-18006}"
export ASR_BACKEND_URL="${ASR_BACKEND_URL:-http://127.0.0.1:18005/v1/audio/transcriptions}"
export YTDLP_BIN="${YTDLP_BIN:-$VENV/bin/yt-dlp}"

# Bearer (clientes vía LiteLLM + llamada al backend): reutiliza la key del ASR batch.
KEY_FILE="${ASR_API_KEY_FILE:-/workspace/asr-api-key}"
if [ -z "${ASR_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
    export ASR_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
fi
# HF token para los modelos gated de diart/pyannote.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/workspace/hf-token}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

exec "$VENV/bin/uvicorn" stream_server:app \
    --app-dir "$SCRIPT_DIR" \
    --host "$ASR_RT_HOST" --port "$ASR_RT_PORT" \
    --workers 1 --timeout-keep-alive 75
