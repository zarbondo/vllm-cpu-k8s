#!/usr/bin/env bash
# End-to-end check of a running deployment (needs `kubectl port-forward svc/vllm 8000:8000`):
#   1. without the Bearer token the API answers 401;
#   2. with it, a chat completion works;
#   3. the pod log has the fact of the request but neither the prompt nor the answer.
set -euo pipefail

NS="${NS:-llm}"
URL="${VLLM_URL:-http://127.0.0.1:8000}"
SECRETS="$(dirname "${BASH_SOURCE[0]}")/../k8s/overlays/minikube/secrets.env"
if [[ -z "${VLLM_API_KEY:-}" && -f "$SECRETS" ]]; then
  VLLM_API_KEY="$(sed -n 's/^VLLM_API_KEY=//p' "$SECRETS")"
fi
: "${VLLM_API_KEY:?set VLLM_API_KEY or create k8s/overlays/minikube/secrets.env}"

fail() { echo "FAIL: $*" >&2; exit 1; }

code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/models")
[[ "$code" == "401" ]] || fail "expected 401 without a token, got $code"
echo "ok   401 without token"

model=$(curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
marker="privacy-check-$RANDOM$RANDOM"
body=$(python3 - "$model" "$marker" <<'EOF'
import json, sys
model, marker = sys.argv[1:]
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": f"Repeat this code word twice: {marker}"}],
    "max_tokens": 32, "temperature": 0,
}))
EOF
)
answer=$(curl -sf -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d "$body" "$URL/v1/chat/completions")
text=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"$answer")
echo "ok   chat completion: ${text:0:80}"

sleep 1
logs=$(kubectl -n "$NS" logs deploy/vllm -c vllm --since=5m)
grep -q "$marker" <<<"$logs" && fail "the prompt marker is in the pod log"
if [[ ${#text} -ge 10 ]] && grep -qF -- "$text" <<<"$logs"; then
  fail "the generated answer is in the pod log"
fi
grep -q "request_received" <<<"$logs" || fail "no request_received line in the pod log"
grep -q "request_finished" <<<"$logs" || fail "no request_finished line in the pod log"
echo "ok   log has the request facts, no prompt:"
grep -E "request_(received|finished)" <<<"$logs" | tail -2
