# Shortcuts for the local minikube flow (README.md has the same steps as plain commands).
SHELL := /bin/bash

VLLM_IMAGE       ?= vllm-cpu:0.32.0
DOWNLOADER_IMAGE ?= vllm-model-downloader:0.1.0
OVERLAY          ?= k8s/overlays/minikube
NS               ?= llm
PYTHON           ?= python3

.PHONY: minikube images secrets deploy wait port-forward logs smoke loadtest test test-image lock undeploy

minikube:            ## local cluster: 6 CPUs, 14 GiB RAM
	minikube start --cpus=6 --memory=14g --disk-size=40g

images:              ## build both images straight into minikube's Docker
	eval $$(minikube docker-env) && \
	docker build -f docker/model-downloader/Dockerfile -t $(DOWNLOADER_IMAGE) . && \
	docker build -f docker/vllm/Dockerfile -t $(VLLM_IMAGE) .

secrets:             ## k8s/overlays/minikube/secrets.env with a random API key
	./scripts/make-secrets.sh

deploy:
	kubectl apply -k $(OVERLAY)

wait:                ## first start: download + load + compile, up to ~15 min
	kubectl -n $(NS) rollout status deploy/vllm --timeout=30m

port-forward:
	kubectl -n $(NS) port-forward svc/vllm 8000:8000

logs:
	kubectl -n $(NS) logs -f deploy/vllm -c vllm

smoke:               ## needs port-forward in another terminal
	./scripts/smoke-test.sh

loadtest:            ## needs port-forward; writes the table into README.md
	VLLM_API_KEY=$$(sed -n 's/^VLLM_API_KEY=//p' k8s/overlays/minikube/secrets.env) \
	$(PYTHON) loadtest/loadtest.py --update-readme README.md

test:                ## host-side tests: download script, manifests, load test
	$(PYTHON) -m pytest -q tests

test-image:          ## privacy logging tests inside the vLLM image (real vLLM code)
	eval $$(minikube docker-env) && \
	tar -C privacy-logging -c tests logging.json | \
	docker run -i --rm --user 0 --entrypoint sh $(VLLM_IMAGE) -c \
	  "mkdir /src && tar -C /src -x && pip install -q pytest && cd /tmp && python -m pytest -q -p no:cacheprovider /src/tests"

lock:                ## re-resolve the pinned requirements
	./scripts/lock-deps.sh

undeploy:            ## keeps nothing: the PVC with the weights is deleted too
	kubectl delete -k $(OVERLAY)
