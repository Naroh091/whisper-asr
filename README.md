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
| `HF_HOME` | — | caché de modelos de Hugging Face |

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
      api_key: "EMPTY"
    model_info:
      mode: audio_transcription
```

## Licencia

MIT (ver `LICENSE`). El modelo Whisper large-v3 tiene su propia licencia (MIT, de OpenAI).
