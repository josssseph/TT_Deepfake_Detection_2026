FROM python:3.11.9-slim-bookworm@sha256:2856e6af199e8128161abd320575eb9b341f3b76f017b5d0c9cd364f60d8a050

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1

LABEL org.opencontainers.image.title="Detector local de Deepfakes"
LABEL org.opencontainers.image.description="Interfaz Streamlit para el modelo espacial-espectral de la tesis"
LABEL org.opencontainers.image.version="1.0.1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock requirements.txt ./
RUN python -m pip install --upgrade pip==26.1.2 \
    && python -m pip install --requirement requirements.lock \
    && python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Valida los recursos propios, descarga ResNet-18 durante el build y ejecuta
# las pruebas antes de producir una imagen utilizable.
RUN python -m deepfake_app.check_assets \
    && python -m pytest -q

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/torch-cache

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
