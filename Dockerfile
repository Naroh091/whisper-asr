# Imagen GPU-ready sin base CUDA pesada: cuBLAS 12 + cuDNN 9 los aportan los
# wheels de NVIDIA (en requirements.txt), y el driver lo da el host vía
# NVIDIA Container Toolkit. Ejecutar con:  docker run --gpus all ...
FROM python:3.12-slim

# ffmpeg para decodificar cualquier contenedor de audio/vídeo
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# venv en ruta conocida; run-asr.sh lo toma vía ASR_VENV y arma LD_LIBRARY_PATH
ENV ASR_VENV=/opt/asr-venv
RUN python -m venv "$ASR_VENV"
COPY requirements.txt .
RUN "$ASR_VENV/bin/pip" install --no-cache-dir --upgrade pip \
    && "$ASR_VENV/bin/pip" install --no-cache-dir -r requirements.txt

# El servicio de streaming usa un entorno separado por la compatibilidad de diart
# con PyTorch/pyannote. El índice CPU evita instalar otra copia de CUDA.
RUN python -m venv /opt/diart-venv
COPY streaming/requirements.txt /tmp/streaming-requirements.txt
RUN /opt/diart-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/diart-venv/bin/pip install --no-cache-dir \
       torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
       --index-url https://download.pytorch.org/whl/cpu \
    && /opt/diart-venv/bin/pip install --no-cache-dir -r /tmp/streaming-requirements.txt

COPY server.py run-asr.sh streaming/ /app/streaming/
RUN chmod +x /app/run-asr.sh /app/streaming/run-stream.sh

ENV ASR_HOST=0.0.0.0 \
    ASR_PORT=18005 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE=float16

EXPOSE 18005 18006

# El modelo se descarga en el primer arranque a HF_HOME; monta un volumen ahí
# (ej. -v whisper-cache:/root/.cache/huggingface) para no rebajarlo cada vez.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:18005/health').status==200 else 1)"

CMD ["./run-asr.sh"]
