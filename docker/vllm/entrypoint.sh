#!/usr/bin/env bash
# Starts the vLLM OpenAI-compatible server on the model from the volume.
# Settings: env (ConfigMap vllm-env) + `vllm serve` flags in $VLLM_SERVE_CONFIG
# (ConfigMap vllm-serve-config). Extra arguments are passed to `vllm serve`.
set -euo pipefail

log() { echo "[entrypoint] $*" >&2; }

: "${MODEL_ID:?MODEL_ID is not set}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}/${MODEL_ID//\//--}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"

# 1. Weights. In k8s the init container has already put them into the volume.
#    MODEL_AUTO_DOWNLOAD=true runs the same script here instead (plain docker run);
#    it returns at once, without network, when the volume already has the model.
if [[ "${MODEL_AUTO_DOWNLOAD:-false}" == "true" ]]; then
  HF_HUB_OFFLINE=0 python /opt/app/download_model.py
fi
if [[ ! -f "${MODEL_DIR}/.download-complete" ]]; then
  log "no complete model in ${MODEL_DIR}: run the model-downloader init container or set MODEL_AUTO_DOWNLOAD=true"
  exit 1
fi

# 2. Clients must authenticate. vLLM reads the key from VLLM_API_KEY (k8s Secret),
#    so it never shows up in the process list.
if [[ -z "${VLLM_API_KEY:-}" && "${ALLOW_UNAUTHENTICATED:-false}" != "true" ]]; then
  log "VLLM_API_KEY is empty, refusing to start an open endpoint (set ALLOW_UNAUTHENTICATED=true to override)"
  exit 1
fi

# 3. OpenMP threads. Without the static CPU manager a pod sees every core of the
#    node while its CFS quota is only limits.cpu, so vLLM's default binding
#    (`auto`, one thread per visible core) oversubscribes the quota. With `nobind`
#    the thread count comes from the limit (CPU_LIMIT, downward API), one core is
#    left for the API server process.
if [[ "${VLLM_CPU_OMP_THREADS_BIND:-auto}" == "nobind" && -z "${OMP_NUM_THREADS:-}" ]]; then
  cpus="${CPU_LIMIT:-$(nproc)}"
  export OMP_NUM_THREADS=$(( cpus > 2 ? cpus - 1 : 1 ))
  log "OMP_NUM_THREADS=${OMP_NUM_THREADS} (cpu limit ${cpus})"
fi

args=("${MODEL_DIR}" --served-model-name "${SERVED_MODEL_NAME}" --host 0.0.0.0 --port "${PORT:-8000}")
if [[ -f "${VLLM_SERVE_CONFIG:-}" ]]; then
  args+=(--config "${VLLM_SERVE_CONFIG}")
fi

log "serving ${MODEL_ID} from ${MODEL_DIR} as '${SERVED_MODEL_NAME}'"
exec vllm serve "${args[@]}" "$@"
