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

# Whisper + diarización corren en CPU para dejar toda la VRAM a Gemma (contexto 256K).
# Esta CPU (Threadripper, AVX-512) transcribe large-v3 int8 a ~6-8x realtime: sobra
# para batch y para el streaming (ventanas <=20s). cpu_threads acotado para no ahogar
# a diart. Volver a GPU: WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 CUDA_VISIBLE_DEVICES=0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"   # vacío = sin GPU (0 VRAM)
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cpu}"
export WHISPER_COMPUTE="${WHISPER_COMPUTE:-int8}"
export WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-1}"
export WHISPER_CPU_THREADS="${WHISPER_CPU_THREADS:-16}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export ASR_HOST="${ASR_HOST:-127.0.0.1}"
export ASR_PORT="${ASR_PORT:-18005}"

# API key opcional: si existe el fichero (fuera del repo), se exige Bearer.
# LiteLLM reenvía su `api_key` como Authorization: Bearer en /audio/transcriptions.
KEY_FILE="${ASR_API_KEY_FILE:-$SCRIPT_DIR/../asr-api-key}"
if [ -z "${ASR_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
    export ASR_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
fi

# Token HF para diarization (pyannote, modelos gated). Si existe el fichero
# (fuera del repo, chmod 600), se carga y se habilita la diarización.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$SCRIPT_DIR/../hf-token}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi
export ASR_ENABLE_DIARIZATION="${ASR_ENABLE_DIARIZATION:-1}"

exec "$VENV/bin/uvicorn" server:app \
    --app-dir "$SCRIPT_DIR" \
    --host "$ASR_HOST" --port "$ASR_PORT" \
    --workers 1 --timeout-keep-alive 75
