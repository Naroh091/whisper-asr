#!/bin/bash
# Wrapper para supervisord: arranca el backend ASR (faster-whisper) en :18005.
# CTranslate2 necesita cuBLAS 12 + cuDNN 9; los servimos desde los wheels de
# NVIDIA instalados en el propio venv (no tocamos el CUDA del sistema de vLLM).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# venv hermano por defecto (en producción: /workspace/asr-venv); override con ASR_VENV.
VENV="${ASR_VENV:-$SCRIPT_DIR/../asr-venv}"

# libs CUDA del venv -> LD_LIBRARY_PATH (cublas, cudnn, nvrtc)
NV="$VENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NV" -name lib -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export WHISPER_COMPUTE="${WHISPER_COMPUTE:-float16}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export ASR_HOST="${ASR_HOST:-127.0.0.1}"
export ASR_PORT="${ASR_PORT:-18005}"

# API key opcional: si existe el fichero (fuera del repo), se exige Bearer.
# LiteLLM reenvía su `api_key` como Authorization: Bearer en /audio/transcriptions.
KEY_FILE="${ASR_API_KEY_FILE:-$SCRIPT_DIR/../asr-api-key}"
if [ -z "${ASR_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
    export ASR_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
fi

exec "$VENV/bin/uvicorn" server:app \
    --app-dir "$SCRIPT_DIR" \
    --host "$ASR_HOST" --port "$ASR_PORT" \
    --workers 1 --timeout-keep-alive 75
