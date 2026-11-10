#!/usr/bin/env bash
# Creates k8s/overlays/minikube/secrets.env with a random API key.
# Usage: ./scripts/make-secrets.sh [hf_token]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=k8s/overlays/minikube/secrets.env
if [[ -f "$out" ]]; then
  echo "$out already exists, not overwriting" >&2
  exit 0
fi
key="$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')"
umask 077
printf 'VLLM_API_KEY=%s\nHF_TOKEN=%s\n' "$key" "${1:-}" > "$out"
echo "written $out"
echo "API key: $key"
