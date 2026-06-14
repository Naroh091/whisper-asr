#!/usr/bin/env python
"""
Backend ASR (speech-to-text) OpenAI-compatible para el stack de inferencia.

Sirve POST /v1/audio/transcriptions igual que la API de OpenAI Whisper, de modo
que LiteLLM (en K8s) lo enruta con provider `openai/` + api_base. Por debajo usa
faster-whisper (CTranslate2, sin torch) sobre Whisper large-v3.

  - Decodifica cualquier formato de entrada (mp3/m4a/webm/ogg/vídeo) vía ffmpeg
    a 16 kHz mono, así no dependemos del decoder interno.
  - VAD activado por defecto para evitar las alucinaciones típicas de Whisper en
    silencios/música.
  - Acceso a GPU serializado con un lock (pensado para compartir GPU con otros
    servicios de inferencia).

Config por entorno (ver run-asr.sh):
  WHISPER_MODEL        modelo CT2 o tamaño        (def. large-v3)
  WHISPER_DEVICE       cuda | cpu                 (def. cuda)
  WHISPER_COMPUTE      float16 | int8_float16 ... (def. float16)
  WHISPER_BEAM_SIZE    beam search                (def. 5)
  ASR_DEFAULT_LANG     idioma forzado por defecto (def. vacío = autodetect)
  ASR_API_KEY          si se define, exige Authorization: Bearer <key> (def. sin auth)
"""
import os
import hmac
import asyncio
import subprocess
import logging

import numpy as np
from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

log = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
DEFAULT_LANG = os.environ.get("ASR_DEFAULT_LANG", "").strip() or None
API_KEY = os.environ.get("ASR_API_KEY", "").strip() or None

app = FastAPI(title="whisper-asr", version="1.0")
_model: WhisperModel | None = None
_gpu_lock = asyncio.Lock()  # serializa el acceso a la GPU (compartida con otros servicios)


def _check_auth(authorization: str | None) -> None:
    """Si ASR_API_KEY está definido, exige Authorization: Bearer <key>.
    El gateway (LiteLLM) reenvía su `api_key` como ese header en la ruta de
    transcripción —es lo único que reenvía—, así que es lo que validamos."""
    if API_KEY is None:
        return
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "API key inválida o ausente")


@app.on_event("startup")
def _load():
    global _model
    log.info("cargando %s en %s (%s)...", MODEL_NAME, DEVICE, COMPUTE)
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)
    log.info("modelo listo")


def _decode_to_pcm(raw: bytes) -> np.ndarray:
    """Cualquier contenedor -> float32 mono 16 kHz vía ffmpeg (lee de stdin)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(400, f"no se pudo decodificar el audio: "
                                 f"{proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _srt_ts(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def _vtt_ts(t: float) -> str:
    return _srt_ts(t).replace(",", ".")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),          # se acepta y se ignora (compat OpenAI)
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    # OpenAI manda los granularities como campos repetidos `timestamp_granularities[]`.
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    if _model is None:
        raise HTTPException(503, "modelo aún no cargado")

    audio = _decode_to_pcm(await file.read())
    lang = (language or DEFAULT_LANG) or None
    # word timestamps si el cliente lo pide explícitamente (granularity "word").
    want_words = "word" in {g.lower() for g in timestamp_granularities}

    async with _gpu_lock:
        segs_gen, info = await asyncio.to_thread(
            lambda: _model.transcribe(
                audio, language=lang, beam_size=BEAM_SIZE,
                temperature=temperature, initial_prompt=prompt,
                vad_filter=True, word_timestamps=want_words,
            )
        )
        segments = await asyncio.to_thread(list, segs_gen)

    text = "".join(s.text for s in segments).strip()
    fmt = (response_format or "json").lower()

    if fmt == "text":
        return PlainTextResponse(text + "\n")
    if fmt == "srt":
        body = "".join(f"{i}\n{_srt_ts(s.start)} --> {_srt_ts(s.end)}\n{s.text.strip()}\n\n"
                       for i, s in enumerate(segments, 1))
        return PlainTextResponse(body, media_type="application/x-subrip")
    if fmt == "vtt":
        body = "WEBVTT\n\n" + "".join(
            f"{_vtt_ts(s.start)} --> {_vtt_ts(s.end)}\n{s.text.strip()}\n\n" for s in segments)
        return PlainTextResponse(body, media_type="text/vtt")
    if fmt == "verbose_json":
        def _words(s):
            return [
                {"word": w.word, "start": round(w.start, 3),
                 "end": round(w.end, 3), "probability": round(w.probability, 4)}
                for w in (s.words or [])
            ]
        payload = {
            "task": "transcribe",
            "language": info.language,
            "duration": round(info.duration, 3),
            "text": text,
            "segments": [
                {"id": i, "start": round(s.start, 3), "end": round(s.end, 3),
                 "text": s.text, "avg_logprob": s.avg_logprob,
                 "no_speech_prob": s.no_speech_prob,
                 **({"words": _words(s)} if want_words else {})}
                for i, s in enumerate(segments)
            ],
        }
        if want_words:
            # OpenAI devuelve además un array `words` aplanado a nivel raíz.
            payload["words"] = [w for s in segments for w in _words(s)]
        return JSONResponse(payload)
    # json (por defecto)
    return JSONResponse({"text": text})
