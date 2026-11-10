"""Static checks of the rendered manifests (needs `kustomize` or `kubectl` in PATH)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ["k8s/overlays/minikube", "k8s/overlays/minikube-small"]
PSA_VOLUME_TYPES = {"configMap", "emptyDir", "persistentVolumeClaim", "secret", "projected", "downwardAPI"}


def render(overlay: str) -> list[dict]:
    secrets = ROOT / "k8s/overlays/minikube/secrets.env"
    created = not secrets.exists()
    if created:
        shutil.copy(secrets.with_name("secrets.env.example"), secrets)
    try:
        if shutil.which("kustomize"):
            cmd = ["kustomize", "build", str(ROOT / overlay)]
        elif shutil.which("kubectl"):
            cmd = ["kubectl", "kustomize", str(ROOT / overlay)]
        else:
            pytest.skip("neither kustomize nor kubectl is installed")
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    finally:
        if created:
            secrets.unlink()
    return [d for d in yaml.safe_load_all(out) if d]


@pytest.fixture(params=OVERLAYS)
def docs(request):
    return render(request.param)


def pod_spec(docs):
    (dep,) = [d for d in docs if d["kind"] == "Deployment"]
    return dep["spec"]["template"]["spec"]


def test_pod_security_restricted(docs):
    spec = pod_spec(docs)
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True and psc["runAsUser"] != 0
    assert psc["seccompProfile"]["type"] == "RuntimeDefault"
    for c in spec["initContainers"] + spec["containers"]:
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False, c["name"]
        assert sc["capabilities"]["drop"] == ["ALL"], c["name"]
        assert sc["readOnlyRootFilesystem"] is True, c["name"]
    for v in spec["volumes"]:
        assert set(v) - {"name"} <= PSA_VOLUME_TYPES, v


def test_secrets_reach_only_their_container(docs):
    spec = pod_spec(docs)
    env = {c["name"]: {e["name"] for e in c.get("env", [])} for c in spec["initContainers"] + spec["containers"]}
    assert "HF_TOKEN" in env["model-downloader"] and "VLLM_API_KEY" not in env["model-downloader"]
    assert "VLLM_API_KEY" in env["vllm"] and "HF_TOKEN" not in env["vllm"]


def test_generated_names_are_wired(docs):
    names = {d["metadata"]["name"] for d in docs if d["kind"] in ("ConfigMap", "Secret")}
    spec = pod_spec(docs)
    refs = set()
    for c in spec["initContainers"] + spec["containers"]:
        refs |= {e["configMapRef"]["name"] for e in c.get("envFrom", [])}
        refs |= {e["valueFrom"]["secretKeyRef"]["name"] for e in c.get("env", []) if "secretKeyRef" in e.get("valueFrom", {})}
    refs |= {v["configMap"]["name"] for v in spec["volumes"] if "configMap" in v}
    assert refs <= names, refs - names


def test_model_volume_is_read_only_for_server(docs):
    spec = pod_spec(docs)
    (server,) = spec["containers"]
    mounts = {m["name"]: m for m in server["volumeMounts"]}
    assert mounts["models"].get("readOnly") is True
    assert all(r["requests"] == r["limits"] for r in [server["resources"]])  # Guaranteed QoS
