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
  WHISPER_CPU_THREADS  hilos CTranslate2 en CPU   (def. 0 = auto)
  ASR_DEFAULT_LANG     idioma forzado por defecto (def. vacío = autodetect)
  ASR_API_KEY          si se define, exige Authorization: Bearer <key> (def. sin auth)

Diarization (opcional, requiere torch + pyannote.audio + modelos gated):
  ASR_ENABLE_DIARIZATION  1 para cargar el pipeline pyannote al arrancar (def. 0)
  ASR_DIARIZATION_MODEL   pipeline de pyannote      (def. pyannote/speaker-diarization-3.1)
  HF_TOKEN                token HF con acceso a los modelos gated de pyannote
"""
import os
import json
import hmac
import asyncio
import subprocess
import logging
from collections import Counter

import numpy as np
from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from faster_whisper import WhisperModel

# torch + pyannote son opcionales: el ASR funciona sin ellos (diarization off).
# Se importan de forma defensiva para no romper el arranque si no están.
try:
    import torch
    from pyannote.audio import Pipeline as _PyannotePipeline
except Exception:  # pragma: no cover - entorno sin torch/pyannote
    torch = None
    _PyannotePipeline = None

log = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "0"))  # 0 = default de CTranslate2
DEFAULT_LANG = os.environ.get("ASR_DEFAULT_LANG", "").strip() or None
API_KEY = os.environ.get("ASR_API_KEY", "").strip() or None
# Latido SSE (segundos) para el modo stream=1: hay que emitir bytes antes de que
# Cloudflare corte por 524 (~100s sin respuesta del origen). 15s deja margen de sobra.
SSE_HEARTBEAT = float(os.environ.get("ASR_SSE_HEARTBEAT", "15"))

ENABLE_DIAR = os.environ.get("ASR_ENABLE_DIARIZATION", "0").strip() in ("1", "true", "yes")
# pyannote.audio 4.x: el pipeline "community-1" empaqueta segmentación + embedding
# en un único repo gated. Override con ASR_DIARIZATION_MODEL (p.ej. speaker-diarization-3.1).
DIAR_MODEL = os.environ.get("ASR_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None

app = FastAPI(title="whisper-asr", version="1.1")
_model: WhisperModel | None = None
_diar_pipe = None  # pyannote Pipeline en GPU, o None si diarization deshabilitada
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
    global _model, _diar_pipe
    log.info("cargando %s en %s (%s)...", MODEL_NAME, DEVICE, COMPUTE)
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE, cpu_threads=CPU_THREADS)
    log.info("modelo ASR listo")
    if not ENABLE_DIAR:
        return
    if _PyannotePipeline is None:
        log.warning("ASR_ENABLE_DIARIZATION=1 pero torch/pyannote.audio no están instalados; diarization OFF")
        return
    if not HF_TOKEN:
        log.warning("ASR_ENABLE_DIARIZATION=1 pero falta HF_TOKEN; diarization OFF")
        return
    log.info("cargando pipeline de diarization %s...", DIAR_MODEL)
    pipe = _PyannotePipeline.from_pretrained(DIAR_MODEL, token=HF_TOKEN)
    if DEVICE.startswith("cuda"):
        pipe.to(torch.device("cuda"))
    _diar_pipe = pipe
    log.info("diarization lista (%s, device=%s)", DIAR_MODEL, DEVICE)


def _diarize(audio: np.ndarray, num_speakers, min_speakers, max_speakers):
    """Corre pyannote sobre el waveform float32 mono 16 kHz; devuelve
    lista de turnos (start, end, speaker_label). Bloqueante: llamar en thread."""
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    kw = {}
    if num_speakers:
        kw["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kw["min_speakers"] = min_speakers
        if max_speakers:
            kw["max_speakers"] = max_speakers
    out = _diar_pipe({"waveform": wav, "sample_rate": 16000}, **kw)
    # pyannote 4.x devuelve DiarizeOutput(speaker_diarization=Annotation, ...);
    # 3.x devolvía el Annotation directamente. Soportamos ambos.
    ann = getattr(out, "speaker_diarization", out)
    return [(t.start, t.end, lbl) for t, _, lbl in ann.itertracks(yield_label=True)]


def _speaker_for(turns, start, end) -> str | None:
    """Hablante del turno que más solapa con [start, end]; si ninguno solapa,
    el más cercano por punto medio."""
    best, best_ov = None, 0.0
    for ts, te, lbl in turns:
        ov = min(end, te) - max(start, ts)
        if ov > best_ov:
            best_ov, best = ov, lbl
    if best is None and turns:
        mid = (start + end) / 2
        best = min(turns, key=lambda x: abs((x[0] + x[1]) / 2 - mid))[2]
    return best


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


def _decode_url_to_pcm(url: str) -> np.ndarray:
    """Descarga y decodifica una URL (http/https; p.ej. S3 presigned) a float32
    mono 16 kHz. ffmpeg lee la URL directamente en streaming: no carga el fichero
    entero en RAM. Solo http(s) para acotar el SSRF; `-reconnect*` da robustez
    ante cortes de red típicos de descargas largas."""
    from urllib.parse import urlparse
    if urlparse(url).scheme not in ("http", "https"):
        raise HTTPException(400, "file_url debe empezar por http:// o https://")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
         "-i", url, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(400, f"no se pudo descargar/decodificar la URL: "
                                 f"{proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _srt_ts(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def _vtt_ts(t: float) -> str:
    return _srt_ts(t).replace(",", ".")


def _sse_event(event: str, data: str) -> bytes:
    """Serializa un evento SSE. `data` puede tener saltos de línea: se parte en
    varias líneas `data:` como exige el protocolo (un `\\n` crudo rompería el evento)."""
    body = "".join(f"data: {ln}\n" for ln in data.split("\n"))
    return f"event: {event}\n{body}\n".encode("utf-8")


async def _sse_stream(work):
    """Envuelve el cómputo (`work`, un coroutine que devuelve un Response) en un
    stream SSE con latidos. Emite `: ping` cada SSE_HEARTBEAT segundos mientras la
    transcripción corre en background, para que Cloudflare no dispare el 524, y al
    terminar un evento `done` con el cuerpo de la respuesta (o `error` si falla).

    Nota: en streaming el status HTTP ya es 200 cuando empieza el trabajo, así que
    los fallos NO pueden viajar como código HTTP; van como evento `error`."""
    yield b": connected\n\n"  # abre la respuesta ya, antes del primer latido
    task = asyncio.ensure_future(work)
    try:
        while True:
            try:
                # shield: el timeout cancela la espera, no el trabajo de fondo.
                resp = await asyncio.wait_for(asyncio.shield(task), SSE_HEARTBEAT)
                break
            except asyncio.TimeoutError:
                yield b": ping\n\n"
        body = bytes(resp.body)
        yield _sse_event("done", body.decode("utf-8", "ignore"))
    except HTTPException as e:
        yield _sse_event("error", json.dumps({"error": {"message": e.detail,
                                                         "status": e.status_code}}))
    except Exception as e:  # pragma: no cover - salvaguarda
        log.exception("fallo en transcripción (stream)")
        yield _sse_event("error", json.dumps({"error": {"message": str(e)}}))
    finally:
        if not task.done():
            task.cancel()  # cliente desconectado: no malgastes CPU/GPU


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE,
            "diarization": _diar_pipe is not None}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile | None = File(default=None),
    # Extensión propia (no OpenAI): en vez de subir el binario, pasar una URL
    # (http/https, S3 presigned…) y el servidor la lee. Evita el límite de tamaño
    # de subida del gateway; ffmpeg la descarga en streaming, sin cargarla en RAM.
    file_url: str | None = Form(default=None),
    model: str = Form(default=MODEL_NAME),          # se acepta y se ignora (compat OpenAI)
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    # OpenAI manda los granularities como campos repetidos `timestamp_granularities[]`.
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
    # Diarización (extensión propia; no es parte de la API de OpenAI). num_speakers
    # fija el nº exacto; si no, se acota con min/max (todos opcionales = autodetect).
    diarization: bool = Form(default=False),
    num_speakers: int | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    # stream=1 -> respuesta SSE con latidos (evita el 524 de Cloudflare en audios
    # largos). Mismo cuerpo final, envuelto en un evento `done`. Ver _sse_stream.
    stream: bool = Form(default=False),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    if _model is None:
        raise HTTPException(503, "modelo aún no cargado")

    fmt = (response_format or "json").lower()
    # Dispara diarización con el flag propio o con el response_format de OpenAI
    # `diarized_json` (este último viaja limpio por LiteLLM al ser param estándar).
    do_diar = diarization or fmt == "diarized_json"
    if do_diar and _diar_pipe is None:
        raise HTTPException(400, "diarization no disponible en este servidor "
                                 "(requiere ASR_ENABLE_DIARIZATION=1 + pyannote + HF_TOKEN)")

    async def _work() -> Response:
        """Todo el cómputo (decode + Whisper + diarización + formato). Devuelve el
        Response final. Se ejecuta directo en modo normal, o envuelto en SSE cuando
        stream=1 (mismo cuerpo, para no duplicar la lógica de formatos)."""
        if file is not None:
            audio = _decode_to_pcm(await file.read())
        elif file_url:
            # descarga+decodifica en un hilo para no bloquear el event loop
            audio = await asyncio.to_thread(_decode_url_to_pcm, file_url)
        else:
            raise HTTPException(400, "falta 'file' (multipart) o 'file_url'")
        lang = (language or DEFAULT_LANG) or None
        # word timestamps si el cliente lo pide, o siempre que haya diarización
        # (los necesitamos para asignar hablante palabra a palabra).
        want_words = "word" in {g.lower() for g in timestamp_granularities} or do_diar

        async with _gpu_lock:
            segs_gen, info = await asyncio.to_thread(
                lambda: _model.transcribe(
                    audio, language=lang, beam_size=BEAM_SIZE,
                    temperature=temperature, initial_prompt=prompt,
                    vad_filter=True, word_timestamps=want_words,
                )
            )
            segments = await asyncio.to_thread(list, segs_gen)
            # Diarización dentro del mismo lock: evita picos simultáneos de VRAM.
            turns = None
            if do_diar:
                turns = await asyncio.to_thread(
                    _diarize, audio, num_speakers, min_speakers, max_speakers)

        text = "".join(s.text for s in segments).strip()

        # Etiqueta de hablante por segmento (mayoría de sus palabras) y por palabra.
        seg_spk: list[str | None] = []
        word_spk: list[list[str | None]] = []
        if turns is not None:
            for s in segments:
                ws = s.words or []
                wspk = [_speaker_for(turns, w.start, w.end) for w in ws]
                word_spk.append(wspk)
                if wspk:
                    seg_spk.append(Counter(wspk).most_common(1)[0][0])
                else:
                    seg_spk.append(_speaker_for(turns, s.start, s.end))

        # Mapeo de etiquetas pyannote (SPEAKER_00, ...) a letras "A","B",... por orden
        # de aparición, como hace OpenAI en diarized_json.
        alpha: dict[str, str] = {}
        if turns is not None:
            for sp in seg_spk:
                if sp and sp not in alpha:
                    n = len(alpha)
                    alpha[sp] = chr(ord("A") + n) if n < 26 else f"S{n}"

        def _pfx(i):  # prefijo "SPEAKER_xx: " para formatos de texto/subtítulos
            return f"{seg_spk[i]}: " if turns is not None and seg_spk[i] else ""

        if fmt == "diarized_json":
            # Esquema de OpenAI (gpt-4o-transcribe-diarize): segmentos con speaker
            # alfabético, texto y tiempos. Es el formato que el cliente pide por LiteLLM.
            return JSONResponse({
                "task": "transcribe",
                "duration": round(info.duration, 3),
                "text": text,
                "segments": [
                    {"id": i, "speaker": alpha.get(seg_spk[i]),
                     "text": s.text.strip(),
                     "start": round(s.start, 3), "end": round(s.end, 3)}
                    for i, s in enumerate(segments)
                ],
            })

        if fmt == "text":
            if turns is None:
                return PlainTextResponse(text + "\n")
            body = "".join(f"{_pfx(i)}{s.text.strip()}\n" for i, s in enumerate(segments))
            return PlainTextResponse(body)
        if fmt == "srt":
            body = "".join(f"{i}\n{_srt_ts(s.start)} --> {_srt_ts(s.end)}\n{_pfx(i-1)}{s.text.strip()}\n\n"
                           for i, s in enumerate(segments, 1))
            return PlainTextResponse(body, media_type="application/x-subrip")
        if fmt == "vtt":
            body = "WEBVTT\n\n" + "".join(
                f"{_vtt_ts(s.start)} --> {_vtt_ts(s.end)}\n{_pfx(i)}{s.text.strip()}\n\n"
                for i, s in enumerate(segments))
            return PlainTextResponse(body, media_type="text/vtt")
        if fmt == "verbose_json":
            def _words(i, s):
                spk = word_spk[i] if turns is not None else None
                out = []
                for j, w in enumerate(s.words or []):
                    d = {"word": w.word, "start": round(w.start, 3),
                         "end": round(w.end, 3), "probability": round(w.probability, 4)}
                    if spk is not None:
                        d["speaker"] = spk[j]
                    out.append(d)
                return out
            payload = {
                "task": "transcribe",
                "language": info.language,
                "duration": round(info.duration, 3),
                "text": text,
                "segments": [
                    {"id": i, "start": round(s.start, 3), "end": round(s.end, 3),
                     "text": s.text, "avg_logprob": s.avg_logprob,
                     "no_speech_prob": s.no_speech_prob,
                     **({"speaker": seg_spk[i]} if turns is not None else {}),
                     **({"words": _words(i, s)} if want_words else {})}
                    for i, s in enumerate(segments)
                ],
            }
            if want_words:
                payload["words"] = [w for i, s in enumerate(segments) for w in _words(i, s)]
            if turns is not None:
                payload["speakers"] = sorted({lbl for _, _, lbl in turns})
            return JSONResponse(payload)
        # json (por defecto). Con diarización devolvemos también segmentos+hablante.
        if turns is not None:
            return JSONResponse({
                "text": text,
                "speakers": sorted({lbl for _, _, lbl in turns}),
                "segments": [
                    {"start": round(s.start, 3), "end": round(s.end, 3),
                     "speaker": seg_spk[i], "text": s.text}
                    for i, s in enumerate(segments)
                ],
            })
        return JSONResponse({"text": text})

    if not stream:
        return await _work()

    # SSE: latidos hasta que _work() termine, luego el cuerpo en un evento `done`.
    # X-Accel-Buffering desactiva el buffering de nginx; los proxies aguas arriba
    # (LiteLLM pass-through) también deben reenviar el stream sin bufferizar.
    return StreamingResponse(
        _sse_stream(_work()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
