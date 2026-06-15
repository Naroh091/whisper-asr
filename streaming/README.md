# Servicio de transcripción + diarización en streaming

Servicio para transcribir y diarizar **directos** (YouTube live, HLS…) en tiempo
casi real. Complementa al backend batch (`../server.py`): aquí va la parte de
**directo**, que es un proceso largo con salida por SSE, no un request/response.

```
POST /v1/realtime/transcriptions   {"source_url": "...", "language": "es"}
  -> text/event-stream con eventos estilo OpenAI:
     data: {"type":"transcript.text.segment","speaker":"A","text":"…","start":12.1,"end":15.4}
```

## Cómo funciona

- **Ingesta**: `yt-dlp` resuelve el directo → `ffmpeg` → PCM 16 kHz mono.
- **Diarización online**: [diart](https://github.com/juanmc2005/diart) (pyannote 3.1)
  en **CPU** — identidad de hablante consistente a lo largo del directo. **0 VRAM**.
- **Transcripción**: NO carga ningún Whisper propio; reutiliza el backend batch
  (`ASR_BACKEND_URL`, p. ej. el `server.py` de este repo con Whisper en GPU) por HTTP.
- **Fusión**: cada ~5 s transcribe la ventana ya estable (con word-timestamps) y
  asigna cada palabra al hablante de diart que la solapa. Latencia ~10–15 s.
- **Memoria acotada**: ventana rodante de `ASR_RT_RETENTION` s (def. 240 = 4 min);
  el audio anterior se descarta.

## Por qué un venv aparte

diart 0.9.x está hecho para **pyannote 3.x** y usa la API vieja `use_auth_token`,
que `huggingface_hub>=0.26` eliminó; pyannote 4.x (el del backend batch) exige el
hub nuevo. No hay versión común → el streaming vive en su propio `diart-venv` con
el stack de la tabla. Ver `requirements.txt` para los pines y el porqué de cada uno.

## Despliegue

```bash
python3 -m venv /workspace/diart-venv
/workspace/diart-venv/bin/pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu
/workspace/diart-venv/bin/pip install -r requirements.txt
apt install -y libportaudio2 ffmpeg      # diart.sources importa sounddevice (PortAudio)
```

Arranque con supervisord: ver `../deploy/asr-stream.conf` (puerto 18006).
Necesita `HF_TOKEN` (modelos gated de pyannote) y, opcionalmente, `ASR_API_KEY`
(Bearer; lo inyecta el gateway). Variables en `run-stream.sh`.

## Exposición (LiteLLM)

Detrás del gateway por **pass-through**. El cliente **debe** enviar `"stream": true`
en el cuerpo, o LiteLLM bufferiza el SSE y no llega nada.
