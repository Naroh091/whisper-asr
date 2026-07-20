# whisper-asr

[![CI](https://github.com/Naroh091/whisper-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/Naroh091/whisper-asr/actions/workflows/ci.yml)

Backend de transcripción de voz (speech-to-text) **OpenAI-compatible**, pensado
para colgar detrás de un gateway LiteLLM. Por debajo usa
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, **sin
torch**) sobre **Whisper large-v3**.

Expone el mismo contrato que la API de OpenAI Whisper, así que cualquier cliente
de transcripción (o LiteLLM con provider `openai/`) funciona apuntando aquí.

## Características

- `POST /v1/audio/transcriptions` — multipart, igual que OpenAI.
- Formatos de salida: `json` (def.), `text`, `verbose_json`, `srt`, `vtt`.
- **VAD activado** por defecto → evita las alucinaciones típicas de Whisper en
  silencios y música de fondo.
- Decodifica cualquier contenedor (mp3, m4a, webm, ogg, vídeo…) vía **ffmpeg** a
  16 kHz mono.
- Acceso a GPU **serializado** con un lock (pensado para compartir GPU con otros
  servicios de inferencia).
- 99 idiomas, incluidos **catalán, gallego y euskera**.

## Requisitos

- Python 3.12
- `ffmpeg` en el `PATH`
- GPU NVIDIA (CUDA 12). Validado en **RTX PRO 6000 Blackwell (sm_120)** con
  CTranslate2 4.7.2 + cuBLAS 12.9 + cuDNN 9.23 (se instalan vía pip, ver abajo).
- También corre en CPU (`WHISPER_DEVICE=cpu`), mucho más lento.

## Instalación

```bash
python3 -m venv asr-venv
asr-venv/bin/pip install -r requirements.txt
```

## Ejecución

```bash
./run-asr.sh         # escucha en 127.0.0.1:18005
```

`run-asr.sh` añade los wheels de cuBLAS/cuDNN del venv a `LD_LIBRARY_PATH` (CTranslate2
los necesita; así no dependemos del CUDA del sistema) y arranca uvicorn.

### Configuración (variables de entorno)

| Variable | Defecto | Descripción |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | modelo CT2 o tamaño |
| `WHISPER_DEVICE` | `cuda` | `cuda` o `cpu` |
| `WHISPER_COMPUTE` | `float16` | `float16`, `int8_float16`, `int8`… |
| `WHISPER_BEAM_SIZE` | `5` | beam search |
| `ASR_DEFAULT_LANG` | (vacío) | idioma forzado por defecto; vacío = autodetect |
| `ASR_API_KEY` | (vacío) | si se define, exige `Authorization: Bearer <key>` (vacío = sin auth) |
| `HF_HOME` | — | caché de modelos de Hugging Face |

### Autenticación

El servicio escucha solo en `127.0.0.1`. Si lo expones a través de un gateway,
define `ASR_API_KEY` y el endpoint de transcripción exigirá `Authorization:
Bearer <key>` (`/health` queda abierto). Nota práctica con **LiteLLM**: en la
ruta `/audio/transcriptions` el proxy **no** reenvía `extra_headers`/`headers`
(probado en 1.85–1.88); lo único que reenvía es `api_key` como `Authorization:
Bearer`. Por eso la autenticación va por `ASR_API_KEY` y no por cabeceras custom.

## Docker

```bash
docker build -t whisper-asr .

# GPU (requiere NVIDIA Container Toolkit). El volumen cachea el modelo entre arranques.
docker run --gpus all -p 18005:18005 \
  -v whisper-cache:/root/.cache/huggingface \
  whisper-asr

# CPU (lento, para pruebas)
docker run -p 18005:18005 -e WHISPER_DEVICE=cpu -e WHISPER_MODEL=base \
  -v whisper-cache:/root/.cache/huggingface \
  whisper-asr
```

## Uso

```bash
curl -s http://127.0.0.1:18005/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3" \
  -F "language=es"
# {"text":"..."}
```

Salud: `GET /health`.

### Streaming SSE (evitar el 524 de Cloudflare)

Audios largos pueden tardar más de los ~100 s que Cloudflare tolera sin respuesta
del origen (error `524`). Con `stream=1` la ruta responde en `text/event-stream`:
emite cabeceras al instante, un evento `segment` por cada segmento según Whisper
lo produce (transcripción visible en vivo), un latido (`: ping`) cada
`ASR_SSE_HEARTBEAT` s (def. 15) en las fases sin segmentos (descarga,
diarización), y al terminar un evento `done` con el **mismo cuerpo** que
devolvería la respuesta normal. Como Cloudflare recibe bytes desde el principio,
no dispara el 524 aunque la transcripción dure minutos.

```bash
curl -N http://127.0.0.1:18005/v1/audio/transcriptions \
  -F "file_url=https://…/audio.mp3" -F "language=es" \
  -F "response_format=diarized_json" -F "diarization=true" \
  -F "stream=1"
# : connected
# : ping           (descarga/decodificación, luego diarización)
# event: segment
# data: {"start":0.0,"end":4.2,"text":"Hola, bienvenidos…","speaker":"A"}
# event: segment
# data: {"start":4.2,"end":9.8,"text":"…","speaker":"B"}
# event: done
# data: {"task":"transcribe",...}
```

Eventos: `segment` (parcial: `start`, `end`, `text` y, con diarización activa,
`speaker`), `done` (data = cuerpo final, JSON en una línea; el cliente debe
quedarse con este y descartar los parciales) o `error` (data =
`{"error":{...}}`). Ojo: en streaming el status HTTP ya es `200` cuando empieza
el trabajo, así que un fallo de decodificación/transcripción **llega como evento
`error`, no como código HTTP**.

Con diarización, pyannote corre **antes** que Whisper (sobre el audio completo:
etiquetas globalmente consistentes, sin renormalizar entre bloques), así que los
primeros `segment` tardan lo que dure esa fase (latidos mientras) pero ya llevan
su hablante: alfabético (`"A"`, `"B"`, …) con `response_format=diarized_json` —
idéntico al mapeo del `done` — o la etiqueta pyannote (`SPEAKER_00`, …) en el
resto de formatos, igual que en el cuerpo final. Sin diarización no hay campo
`speaker` y los `segment` empiezan en cuanto acaba la descarga.

**Importante:** todo proxy intermedio debe reenviar el stream sin bufferizar. La
respuesta ya manda `X-Accel-Buffering: no` (nginx).

### Ruta JSON `/v1/file/transcriptions` (obligatoria tras el pass-through de LiteLLM)

El pass-through de LiteLLM solo abre el upstream en modo streaming si detecta
`stream` en un cuerpo **JSON**; con `multipart/form-data` no parsea el cuerpo,
así que `-F "stream=1"` le resulta invisible y **bufferiza la respuesta SSE
entera** (el cliente no ve ni `: connected` hasta que acaba la transcripción, y
Cloudflare corta con 524). Verificado en el código de LiteLLM hasta v1.94: la
rama multipart siempre lee la respuesta completa.

Por eso existe `POST /v1/file/transcriptions`: mismos campos que la ruta form
pero en JSON, solo con `file_url` (sin subida de fichero). Con `"stream": true`
LiteLLM sí toma su rama streaming y reenvía el SSE chunk a chunk:

```bash
curl -N https://llm.botalite.es/v1/file/transcriptions \
  -H "Authorization: Bearer sk-…" -H "Content-Type: application/json" \
  -d '{"file_url":"https://…/audio.mp3","stream":true,
       "response_format":"diarized_json","diarization":true}'
```

En LiteLLM se configura como Pass Through Endpoint: `/v1/file/transcriptions` →
`http(s)://<asr>/v1/file/transcriptions`. Los campos desconocidos que LiteLLM
inyecta en el JSON se ignoran.

## Despliegue (supervisord)

`deploy/asr.conf` es un ejemplo de programa para supervisord. El servicio escucha
solo en `127.0.0.1`; expónlo a través de un proxy/tunnel con autenticación.

## Integración con LiteLLM

```yaml
model_list:
  - model_name: whisper-large-v3
    litellm_params:
      model: openai/whisper-large-v3
      api_base: http://127.0.0.1:18005/v1
      api_key: "EMPTY"          # si activas ASR_API_KEY, pon aquí ese mismo valor:
                                # LiteLLM lo reenvía como Authorization: Bearer.
    model_info:
      mode: audio_transcription
```

## Licencia

MIT (ver `LICENSE`). El modelo Whisper large-v3 tiene su propia licencia (MIT, de OpenAI).
