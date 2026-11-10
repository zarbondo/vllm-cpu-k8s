# Init container image: puts model weights from Hugging Face into the volume.
# Python + huggingface_hub only (~70 MB), runs as an unprivileged user.
#
#   docker build -f docker/model-downloader/Dockerfile -t vllm-model-downloader:0.1.0 .
#
# Build context is the repository root.

ARG PYTHON_IMAGE=python:3.12-alpine
FROM ${PYTHON_IMAGE}

# Hash-locked wheels only: no sdist builds, no unpinned transitive packages.
COPY docker/model-downloader/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check --require-hashes \
        --only-binary :all: -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && addgroup -S -g 10001 app \
 && adduser -S -D -H -u 10001 -G app app

COPY --chmod=0755 scripts/download_model.py /usr/local/bin/download_model.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODELS_ROOT=/models

USER 10001:10001
ENTRYPOINT ["python", "/usr/local/bin/download_model.py"]
