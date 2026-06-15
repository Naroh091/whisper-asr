#!/usr/bin/env python
"""
Servicio de transcripción + diarización en streaming (directos: YouTube live, HLS…).

Expone POST /v1/realtime/transcriptions con salida text/event-stream (SSE) con la
forma de eventos de OpenAI (`transcript.text.segment` con `speaker`). Diseñado para
enchufarse detrás del pass-through de LiteLLM.

Arquitectura (corre en diart-venv, CPU; 0 VRAM):
  - Ingesta: yt-dlp resuelve el directo -> ffmpeg -> PCM f32 16k mono.
  - Diarización ONLINE: diart (pyannote 3.1, CPU) -> turnos de hablante incrementales
    con identidad consistente a lo largo del directo.
  - Transcripción: se reutiliza el Whisper large-v3 que ya está en GPU (backend batch
    :18005) vía HTTP; diart NO carga ningún whisper.
  - Fusión: cada ~FLUSH s se transcribe la ventana ya "estable" (latest - LATENCY) con
    word-timestamps y se asigna cada palabra al hablante de diart que la solapa.

Config por entorno (ver run-stream.sh):
  ASR_API_KEY            Bearer exigido a los clientes (lo inyecta LiteLLM). Opcional.
  ASR_BACKEND_URL        endpoint de transcripción batch (def. http://127.0.0.1:18005/v1/audio/transcriptions)
  ASR_BACKEND_KEY        Bearer para llamar al backend (def. = ASR_API_KEY)
  HF_TOKEN               token HF para los modelos gated de diart/pyannote
  ASR_RT_LATENCY         latencia de diart en s, mayor = más preciso (def. 5)
  ASR_RT_FLUSH           cada cuántos s se vuelca una ventana (def. 5)
  ASR_RT_MAXSPAN         tamaño máximo de ventana a transcribir de una vez (def. 20)
"""
import os
import json
import struct
import asyncio
import logging
import threading
import subprocess

import numpy as np
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference
from pyannote.core import Annotation

log = logging.getLogger("asr-stream")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = (os.environ.get("ASR_API_KEY") or "").strip() or None
BACKEND_URL = os.environ.get("ASR_BACKEND_URL", "http://127.0.0.1:18005/v1/audio/transcriptions")
BACKEND_KEY = (os.environ.get("ASR_BACKEND_KEY") or os.environ.get("ASR_API_KEY") or "").strip() or None
SAMPLE_RATE = 16000
LATENCY = float(os.environ.get("ASR_RT_LATENCY", "5"))
FLUSH = float(os.environ.get("ASR_RT_FLUSH", "5"))
MINSPAN = 3.0
MAXSPAN = float(os.environ.get("ASR_RT_MAXSPAN", "20"))
# Ventana rodante: se descarta el audio (y los turnos) anteriores a estos segundos
# para acotar la RAM en directos largos. 240s = 4 min.
RETENTION_S = float(os.environ.get("ASR_RT_RETENTION", "240"))

YTDLP = os.environ.get("YTDLP_BIN", "yt-dlp")
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")

app = FastAPI(title="asr-stream", version="1.0")


def _check_auth(authorization: str | None) -> None:
    if API_KEY is None:
        return
    import hmac
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(401, "API key inválida o ausente")


# ---------- estado compartido entre el hilo de diart y el worker async ----------
class StreamState:
    def __init__(self):
        self.audio = bytearray()       # PCM f32 de la ventana retenida (para cortar)
        self.base = 0.0                # segundos ya descartados del inicio de self.audio
        self.diar = Annotation()       # diarización acumulada (podada a la ventana)
        self.latest_time = 0.0         # segundo de audio procesado por diart
        self.lock = threading.Lock()
        self.closed = False
        self.error: BaseException | None = None


# ---------- fuente de audio en vivo para diart (yt-dlp -> ffmpeg) ----------
class FFmpegLiveSource(AudioSource):
    def __init__(self, url: str, state: StreamState, sample_rate: int = SAMPLE_RATE,
                 block_duration: float = 0.5):
        super().__init__(uri="live", sample_rate=sample_rate)
        self.url = url
        self.state = state
        self.block_size = int(round(block_duration * sample_rate))
        self.is_closed = False
        self._yt = None
        self._ff = None

    def _spawn(self):
        # yt-dlp escribe el mejor audio a stdout; ffmpeg lo normaliza a f32 mono 16k.
        self._yt = subprocess.Popen(
            [YTDLP, "-q", "--no-warnings", "-f", "bestaudio/best", "-o", "-", self.url],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._ff = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", str(self.sample_rate), "pipe:1"],
            stdin=self._yt.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if self._yt.stdout:
            self._yt.stdout.close()  # ffmpeg es dueño del pipe ahora

    def read(self):
        nbytes = self.block_size * 4  # float32
        try:
            self._spawn()
            while not self.is_closed and not self.state.closed:
                raw = self._ff.stdout.read(nbytes)
                if not raw or len(raw) < nbytes:
                    break
                with self.state.lock:
                    self.state.audio.extend(raw)
                chunk = np.frombuffer(raw, dtype=np.float32).reshape(1, -1)
                self.stream.on_next(chunk)
        except BaseException as e:  # noqa: BLE001
            self.state.error = e
            self.stream.on_error(e)
        finally:
            self.stream.on_completed()
            self.close()

    def close(self):
        self.is_closed = True
        for p in (self._ff, self._yt):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass


def _make_hook(state: StreamState):
    def hook(result):
        pred, wav = result
        with state.lock:
            for seg, track, label in pred.itertracks(yield_label=True):
                state.diar[seg, track] = label
            try:
                state.latest_time = max(state.latest_time, float(wav.extent.end))
            except Exception:
                pass
    return hook


def _run_diart(state: StreamState, url: str):
    try:
        # Defaults de diart (pyannote/segmentation + pyannote/embedding). Latencia alta
        # = más precisión de hablante. Si la versión no acepta el kwarg, usa el default.
        try:
            cfg = SpeakerDiarizationConfig(latency=LATENCY)
        except TypeError:
            cfg = SpeakerDiarizationConfig()
        pipeline = SpeakerDiarization(cfg)
        source = FFmpegLiveSource(url, state)
        inf = StreamingInference(pipeline, source, do_profile=False, show_progress=False)
        inf.attach_hooks(_make_hook(state))
        log.info("diart arrancado para %s", url)
        inf()
    except BaseException as e:  # noqa: BLE001
        state.error = e
        log.exception("diart terminó con error")
    finally:
        state.closed = True
        log.info("diart finalizado")


# ---------- fusión hablante (diart) + texto (whisper GPU) ----------
def _speaker_for(diar: Annotation, start: float, end: float):
    best, best_ov = None, 0.0
    for seg, _, label in diar.itertracks(yield_label=True):
        ov = min(end, seg.end) - max(start, seg.start)
        if ov > best_ov:
            best_ov, best = ov, label
    return best


def _wav16(pcm_f32: bytes, sr: int) -> bytes:
    arr = np.frombuffer(pcm_f32, dtype=np.float32)
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    n = len(pcm16)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE" + b"fmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" +
           struct.pack("<I", n))
    return hdr + pcm16


async def _transcribe_words(client: httpx.AsyncClient, pcm_f32: bytes, language: str | None):
    if len(pcm_f32) < 4 * SAMPLE_RATE // 2:  # < 0.5s, no merece la pena
        return []
    wav = _wav16(pcm_f32, SAMPLE_RATE)
    data = {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
    if language:
        data["language"] = language
    headers = {"Authorization": f"Bearer {BACKEND_KEY}"} if BACKEND_KEY else {}
    r = await client.post(BACKEND_URL, files={"file": ("slice.wav", wav, "audio/wav")},
                          data=data, headers=headers, timeout=180)
    r.raise_for_status()
    j = r.json()
    if j.get("words"):
        return j["words"]
    # fallback: derivar de los segmentos si el backend no diera words top-level
    out = []
    for s in j.get("segments", []):
        for w in s.get("words", []):
            out.append(w)
    return out


def _alpha(label, mapping: dict):
    if label is None:
        return None
    if label not in mapping:
        n = len(mapping)
        mapping[label] = chr(ord("A") + n) if n < 26 else f"S{n}"
    return mapping[label]


def _fuse(words, diar: Annotation, offset: float, mapping: dict):
    segs, cur = [], None
    for w in words:
        gs, ge = offset + w["start"], offset + w["end"]
        label = _speaker_for(diar, gs, ge)
        # palabra sin solape con ningún turno: arrastra el hablante en curso
        # (evita segmentos sueltos con speaker null entre medias de un turno).
        letter = _alpha(label, mapping) if label is not None else (cur["speaker"] if cur else None)
        if cur and cur["speaker"] == letter:
            cur["text"] += w["word"]
            cur["end"] = ge
        else:
            if cur:
                segs.append(cur)
            cur = {"speaker": letter, "text": w["word"], "start": gs, "end": ge}
    if cur:
        segs.append(cur)
    for s in segs:
        s["text"] = s["text"].strip()
        s["start"] = round(s["start"], 3)
        s["end"] = round(s["end"], 3)
    return [s for s in segs if s["text"]]


def _prune_diar(diar: Annotation, t0: float) -> Annotation:
    """Descarta turnos que terminan antes de t0 (ventana rodante: mantiene la
    diarización acotada en directos largos)."""
    out = Annotation(uri=diar.uri)
    for seg, track, label in diar.itertracks(yield_label=True):
        if seg.end >= t0:
            out[seg, track] = label
    return out


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _flush_worker(state: StreamState, q: asyncio.Queue, language: str | None):
    mapping: dict = {}
    emitted = 0.0
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(FLUSH)
            with state.lock:
                base = state.base
                audio_end = base + len(state.audio) // 4 / SAMPLE_RATE
                latest = state.latest_time
                closed = state.closed
            # nunca por detrás del audio ya descartado por la ventana rodante
            emitted = max(emitted, base)
            # zona "estable" para diarizar: lo procesado menos la latencia de diart
            committed = audio_end if closed else max(0.0, min(latest, audio_end) - LATENCY)
            # vuelca en trozos de hasta MAXSPAN mientras haya margen suficiente
            while committed - emitted >= (0.1 if closed else MINSPAN):
                end = min(committed, emitted + MAXSPAN)
                with state.lock:
                    b = state.base
                    i0 = max(0, int((emitted - b) * SAMPLE_RATE) * 4)
                    i1 = max(i0, int((end - b) * SAMPLE_RATE) * 4)
                    sl = bytes(state.audio[i0:i1])
                    diar = state.diar.copy()
                try:
                    words = await _transcribe_words(client, sl, language)
                    for seg in _fuse(words, diar, emitted, mapping):
                        await q.put(_sse({"type": "transcript.text.segment", **seg}))
                except Exception as e:  # noqa: BLE001
                    await q.put(_sse({"type": "error", "message": str(e)[:300]}))
                emitted = end
            # ventana rodante: descarta audio y turnos anteriores a RETENTION_S.
            with state.lock:
                new_base = max(state.base, state.latest_time - RETENTION_S)
                drop = int((new_base - state.base) * SAMPLE_RATE) * 4
                if drop > 0:
                    del state.audio[:drop]
                    state.base = new_base
                    state.diar = _prune_diar(state.diar, new_base)
            if closed and committed - emitted < 0.1:
                break
    await q.put(None)


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND_URL, "diarization": "diart"}


@app.post("/v1/realtime/transcriptions")
async def realtime(request: Request, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "se espera un cuerpo JSON con source_url")
    url = (body.get("source_url") or body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "falta source_url")
    language = (body.get("language") or "").strip() or None

    state = StreamState()
    q: asyncio.Queue = asyncio.Queue()
    threading.Thread(target=_run_diart, args=(state, url), daemon=True).start()
    worker = asyncio.create_task(_flush_worker(state, q, language))

    async def gen():
        yield _sse({"type": "transcript.session.created", "source_url": url})
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
                if await request.is_disconnected():
                    break
        finally:
            state.closed = True
            worker.cancel()
            if state.error:
                yield _sse({"type": "error", "message": str(state.error)[:300]})
            yield _sse({"type": "transcript.session.done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
