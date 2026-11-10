# vLLM 0.32.0 for CPU (x86_64, AVX512 or AVX2), running as an unprivileged user.
#
#   docker build -f docker/vllm/Dockerfile -t vllm-cpu:0.32.0 .
#
# Build context is the repository root.

# Debian 13 (trixie): glibc 2.41 - the vLLM CPU wheel is manylinux_2_39 - and gcc 14,
# which torch.compile (inductor) uses at runtime to build CPU kernels.
# For a bit-for-bit reproducible base, pin it by digest: python:3.12-slim-trixie@sha256:...
ARG PYTHON_IMAGE=python:3.12-slim-trixie

############################## build ##############################
FROM ${PYTHON_IMAGE} AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Exactly the locked set: --no-deps forbids pip to resolve anything on its own,
# `pip check` proves the set is complete and consistent.
COPY docker/vllm/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r /tmp/requirements.txt \
 && pip check

# Privacy logging plugin (registered through the vllm.general_plugins entry point)
COPY privacy-logging /tmp/privacy-logging
RUN pip install --no-deps --no-build-isolation --no-cache-dir /tmp/privacy-logging

############################# runtime #############################
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="vllm-cpu" \
      org.opencontainers.image.description="vLLM 0.32.0 OpenAI-compatible server, CPU-only, rootless, no prompts in logs"

# libnuma1                - vLLM CPU kernels
# libtcmalloc-minimal4t64 - allocator recommended by vLLM for CPU (see LD_PRELOAD below)
# g++                     - torch.compile/inductor compiles C++ kernels on start-up
# tini                    - PID 1: forwards signals, reaps vLLM worker processes
# System packages follow the Debian stable release of the base image; Python
# packages, which define the behaviour of the server, are pinned exactly.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends libnuma1 libtcmalloc-minimal4t64 g++ tini \
 && rm -rf /var/lib/apt/lists/*

# /models is owned by the user, so a fresh Docker named volume mounted there is writable
# (in k8s the PVC is made writable by fsGroup instead).
ARG UID=10001
RUN groupadd --system --gid ${UID} vllm \
 && useradd --system --uid ${UID} --gid vllm --home-dir /home/vllm --create-home \
            --shell /usr/sbin/nologin vllm \
 && install -d -o ${UID} -g ${UID} /models

# Everything below is owned by root and read-only for the vllm user.
COPY --from=build /opt/venv /opt/venv
COPY privacy-logging/logging.json /etc/vllm/logging.json
COPY k8s/base/config/vllm-serve.yaml /etc/vllm/serve/config.yaml
COPY --chmod=0755 scripts/download_model.py /opt/app/download_model.py
COPY --chmod=0755 docker/vllm/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so

# Every writable path lives under /home/vllm (an emptyDir in k8s), so the rest of
# the filesystem can be mounted read-only.
ENV HOME=/home/vllm \
    XDG_CACHE_HOME=/home/vllm/.cache \
    XDG_CONFIG_HOME=/home/vllm/.config \
    VLLM_CACHE_ROOT=/home/vllm/.cache/vllm \
    VLLM_CONFIG_ROOT=/home/vllm/.config/vllm \
    TORCHINDUCTOR_CACHE_DIR=/home/vllm/.cache/torchinductor \
    TRITON_HOME=/home/vllm \
    NUMBA_CACHE_DIR=/home/vllm/.cache/numba \
    HF_HOME=/home/vllm/.cache/huggingface

# Serving never goes to the network (only the downloader does), no usage stats.
# The privacy patch covers the Python frontend, so the experimental Rust one stays off.
ENV HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_USE_RUST_FRONTEND=0 \
    VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging.json \
    VLLM_SERVE_CONFIG=/etc/vllm/serve/config.yaml \
    MODELS_ROOT=/models

USER ${UID}:${UID}
WORKDIR /home/vllm
EXPOSE 8000

HEALTHCHECK --start-period=10m --interval=30s --timeout=5s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
