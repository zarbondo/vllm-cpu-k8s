# vLLM на CPU в Kubernetes

Инференс `Qwen/Qwen2.5-1.5B-Instruct` через vLLM 0.32.0 (OpenAI-совместимый API) в minikube, строго на CPU.
Образ rootless, все Python-зависимости запинены, промпты и ответы модели не попадают в логи,
веса один раз скачиваются в PVC отдельным init-контейнером, доступ к API по Bearer-токену.

Сделаны все три бонуса: kustomize (`kubectl apply -k`), init-контейнер на отдельном лёгком rootless-образе,
нагрузочный тест с таблицей и разбором настроек.

```
Pod vllm  (namespace llm, Pod Security "restricted", uid 10001)
├─ init: model-downloader   образ vllm-model-downloader: alpine + huggingface_hub
│    env:  ConfigMap vllm-env (модель, ревизия) + HF_TOKEN из Secret
│    PVC /models (rw): маркер есть → выход без сети; нет → скачать, проверить, записать маркер
└─ vllm                     образ vllm-cpu: vLLM 0.32.0 CPU + плагин privacy-logging
     env:  ConfigMap vllm-env + VLLM_API_KEY из Secret; флаги vllm serve из ConfigMap vllm-serve-config
     PVC /models (ro), HF_HUB_OFFLINE=1 → vllm serve :8000
Service vllm:8000  ←  клиенты с "Authorization: Bearer <VLLM_API_KEY>"
```

## Структура

| Путь | Что там |
|---|---|
| `docker/vllm/Dockerfile` | rootless-образ vLLM для CPU (multi-stage, uid 10001) |
| `docker/vllm/requirements.in` → `requirements.txt` | исходная зависимость и полный lock (152 пакета, включая транзитивные) |
| `docker/vllm/entrypoint.sh` | запуск `vllm serve`: модель из тома, флаги из ConfigMap, потоки OpenMP |
| `docker/model-downloader/` | лёгкий rootless-образ init-контейнера (alpine, hash-pinned зависимости) |
| `privacy-logging/` | плагин vLLM + фильтр логов + `logging.json`: логируется факт запроса, без текста |
| `scripts/download_model.py` | скрипт первичного скачивания модели в PVC |
| `scripts/lock-deps.sh` | перегенерация lock-файлов |
| `scripts/make-secrets.sh`, `scripts/smoke-test.sh` | генерация секретов; e2e-проверка токена и логов |
| `k8s/base/` | Namespace, ServiceAccount, PVC, Deployment, Service, генераторы ConfigMap |
| `k8s/overlays/minikube/` | base + Secret из локального `secrets.env` |
| `k8s/overlays/minikube-small/` | то же для слабой машины: Qwen2.5-0.5B, 2 CPU, 6 GiB |
| `loadtest/loadtest.py` | нагрузочный тест: TTFT, latency, throughput |
| `tests/`, `privacy-logging/tests/` | тесты скрипта скачивания, манифестов, нагрузочного теста и логирования |

## Запуск в minikube

Нужны Docker (BuildKit, по умолчанию с Docker 23+), minikube, kubectl и x86_64 CPU с AVX2 или AVX512
(vLLM CPU работает с обоими, AVX512/AMX быстрее). Под модель 1.5B нужно ~10 GiB RAM в minikube,
для слабой машины есть overlay `minikube-small` (~6 GiB). Все команды из корня репозитория,
на Windows удобнее всего из WSL2 (см. ниже), там же работает `make`.

```bash
# 1. кластер
minikube start --cpus=6 --memory=14g --disk-size=40g

# 2. образы собираются прямо в Docker внутри minikube, registry не нужен
eval $(minikube docker-env)
docker build -f docker/model-downloader/Dockerfile -t vllm-model-downloader:0.1.0 .
docker build -f docker/vllm/Dockerfile -t vllm-cpu:0.32.0 .

# 3. секреты: случайный API-ключ (+ HF-токен аргументом, если модель gated)
./scripts/make-secrets.sh            # пишет k8s/overlays/minikube/secrets.env (в .gitignore)

# 4. деплой и ожидание: скачивание ~3 GiB, загрузка и torch.compile занимают несколько минут
kubectl apply -k k8s/overlays/minikube
kubectl -n llm rollout status deploy/vllm --timeout=30m
kubectl -n llm logs deploy/vllm -c model-downloader   # что сделал init-контейнер

# 5. доступ
kubectl -n llm port-forward svc/vllm 8000:8000 &
export VLLM_API_KEY=$(sed -n 's/^VLLM_API_KEY=//p' k8s/overlays/minikube/secrets.env)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct", "messages": [{"role": "user", "content": "Привет! Кто ты?"}]}'

# 6. проверка: 401 без токена, ответ с токеном, в логе пода нет ни промпта, ни ответа
./scripts/smoke-test.sh
```

То же короче: `make minikube images secrets deploy wait`, потом `make port-forward` и `make smoke`.
Удалить всё, включая PVC с весами: `kubectl delete -k k8s/overlays/minikube`.

Без Kubernetes образ запускается и так (скачивание тогда делает entrypoint тем же скриптом):

```bash
docker run --rm -p 8000:8000 --shm-size=1g -v vllm-models:/models \
  -e MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct -e MODEL_AUTO_DOWNLOAD=true -e VLLM_API_KEY=secret \
  vllm-cpu:0.32.0
```

## Как сделано

### Образ: rootless и воспроизводимый

- Multi-stage: в build-стадии ставится venv, в runtime копируется только `/opt/venv`. База
  `python:3.12-slim-trixie`: официальный CPU-wheel vLLM собран под `manylinux_2_39` (нужен glibc ≥ 2.39,
  в Debian 13 он 2.41), а `g++` нужен в рантайме, потому что `torch.compile` (inductor) собирает
  C++-ядра для CPU при старте.
- Ставится официальная CPU-сборка `vllm-0.32.0+cpu` из GitHub-релиза, `torch==2.13.0+cpu` из
  индекса PyTorch для CPU (без CUDA-библиотек), `triton-cpu`. Аллокатор tcmalloc и Intel OpenMP подгружаются
  через `LD_PRELOAD`, как рекомендует документация vLLM для CPU.
- **Зависимости.** `requirements.txt` содержит все 152 пакета с точными версиями, включая транзитивные.
  Установка идёт через `pip install --no-deps`, то есть pip ничего не резолвит сам и в образ не попадёт
  ничего вне lock-файла, а `pip check` в конце сборки валит её, если набор неполный или противоречивый.
  Lock генерируется `scripts/lock-deps.sh` (uv, `--torch-backend cpu`) из `requirements.in`.
  Все пакеты с PyPI проверены на наличие колёс под cp312/manylinux x86_64, то есть при сборке ничего не
  компилируется из исходников. Образ init-контейнера закреплён ещё строже: `--require-hashes`, только wheels.
- **Rootless.** Пользователь `vllm` с uid/gid 10001, `USER 10001:10001` (числовой, чтобы kubelet мог
  проверить `runAsNonRoot`). Код, venv и конфиги принадлежат root и для процесса только читаемы.
  Всё, что vLLM пишет (кэши torch.compile, triton, numba), направлено в `/home/vllm`, поэтому в k8s корневая
  ФС смонтирована read-only, а `/home/vllm`, `/tmp` и `/dev/shm` это `emptyDir`.
- В рантайме сеть не нужна: `HF_HUB_OFFLINE=1`, отключена телеметрия vLLM (`VLLM_NO_USAGE_STATS`).
  PID 1 это `tini`: пробрасывает сигналы и собирает процессы воркеров vLLM.

### Логирование без промптов

Что пишет «голый» vLLM (проверено в `privacy-logging/tests`, тест `test_baseline_vllm_leaks`):
на INFO с `--enable-log-outputs` в лог уходит сгенерированный текст, в параметрах запроса видны
stop-строки; на DEBUG добавляются промпт, его token ids и весь JSON тела запроса; переменная
`VLLM_DEBUG_LOG_API_SERVER_RESPONSE` включает middleware, который логирует тела ответов.

Защита в два независимых слоя (`privacy-logging/`, ставится в образ как пакет):

1. **Плагин vLLM** (`vllm_privacy_logging.plugin`, entry point `vllm.general_plugins`, vLLM грузит его сам
   во всех своих процессах). Он подменяет методы `RequestLogger`: `log_inputs`, `log_outputs`,
   `log_request_body`. Новые версии пишут только метаданные: id запроса, число токенов промпта и ответа,
   числовые параметры сэмплинга, finish reason и длительность. Текст до записи лога просто не доходит,
   поэтому уровень логирования и форматтеры ничего не меняют. Response-logging middleware плагин
   принудительно выключает.
2. **Фильтр логов** (`PrivacyFilter`, подключён через `VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging.json`
   на все хендлеры, включая uvicorn). Отбрасывает записи логгеров, которые несут контент, если они не
   помечены плагином, и DEBUG-записи модулей, работающих с сырым текстом (entrypoints, tool/reasoning parsers).
   У обработчиков ошибок API срезает traceback (с `--log-error-stack` текст ошибки валидации pydantic
   может цитировать поля запроса), строка с id запроса остаётся. Работает и без плагина, например если его отключить через
   `VLLM_PLUGINS`.

Факт обращения логируется всегда, в ConfigMap включены `enable-log-requests` и `enable-log-outputs`
(после патча они безопасны). Так выглядят строки в выводе тестов: две строки плагина на запрос
и access log uvicorn через тот же `logging.json`:

```
(MainProcess pid=1429) INFO 11-10 13:52:06 [plugin.py:130] request_received id=chatcmpl-1 prompt_tokens=3 max_tokens=16 temperature=0.5 top_p=1.0 top_k=0 n=1
(MainProcess pid=1429) INFO 11-10 13:52:06 [plugin.py:169] request_finished id=chatcmpl-1 output_tokens=3 finish_reason=stop stream=true duration_ms=0
(MainProcess pid=2407) INFO 11-10 14:12:00 [httptools_impl.py:482] 127.0.0.1:37158 - "POST /v1/chat/completions HTTP/1.1" 200
```

Без плагина на DEBUG те же вызовы дают `prompt: 'SECRET_PROMPT'`, `output: 'SECRET_OUTPUT_2'` и тело
запроса целиком.

Тесты прогоняют настоящий код vLLM 0.32.0 в отдельных процессах (как это делает сервер) для INFO и DEBUG,
для «только плагин», «только фильтр» и для uvicorn access log: `make test-image` запускает их внутри образа.
В рабочем кластере то же проверяет `scripts/smoke-test.sh`: шлёт промпт с уникальной меткой и ищет её в
`kubectl logs`. Экспериментальный Rust-фронтенд vLLM выключен (`VLLM_USE_RUST_FRONTEND=0`), патч покрывает
Python-фронтенд.

### Кэширование модели в PVC

`scripts/download_model.py` в init-контейнере:

- если в `/models/<org>--<name>/` есть маркер `.download-complete` для тех же `MODEL_ID` и `MODEL_REVISION`,
  он выходит за секунду и в сеть не ходит. Так проходят все рестарты пода;
- иначе резолвит ревизию в commit sha (все файлы из одного коммита), качает только safetensors, конфиги
  и токенизатор (`snapshot_download`, докачка после обрыва), проверяет, что на месте config, токенизатор и
  все шарды из `model.safetensors.index.json`, и атомарно пишет маркер. Упал на середине, значит маркера нет и
  следующий старт докачает;
- сменили модель или ревизию в ConfigMap: старая папка удаляется, качается новая;
- файловый лок на томе не даёт двум подам качать в одну папку одновременно.

Основной контейнер монтирует том read-only, работает с `HF_HUB_OFFLINE=1` и без маркера не стартует.
Логика покрыта тестами (`tests/test_download_model.py`, сеть замокана), включая «повторный старт не трогает сеть».

### Секреты

Secret `vllm-secrets` генерируется kustomize из `k8s/overlays/minikube/secrets.env` (он в `.gitignore`,
в репозитории только `secrets.env.example`). В base Secret намеренно нет: каждое окружение приносит свой
(в проде это был бы External Secrets/Sealed Secrets). Каждый ключ получает только тот контейнер, которому он нужен:

- `HF_TOKEN` только init-контейнеру (`optional: true`, Qwen2.5 публичная);
- `VLLM_API_KEY` только серверу, через переменную окружения, которую vLLM читает сам, поэтому ключа нет в
  аргументах процесса. Все `/v1/*` требуют `Authorization: Bearer <key>`, `/health` открыт для проб.
  Если ключ пустой, entrypoint отказывается стартовать с открытым API.

### Конфигурация через ConfigMap

- `k8s/base/config/vllm.env` → ConfigMap `vllm-env`: модель, ревизия, имя модели в API, режим потоков OpenMP.
- `k8s/base/config/vllm-serve.yaml` → ConfigMap `vllm-serve-config`, монтируется как
  `/etc/vllm/serve/config.yaml` и передаётся в `vllm serve --config`: любые флаги vLLM (dtype, длина контекста,
  батчинг, размер KV-кэша, логирование).

Поменять настройку = отредактировать файл и сделать `kubectl apply -k`. ConfigMap'ы создаются генератором
kustomize с хэшем содержимого в имени, поэтому Deployment выкатывается сам, пересобирать образ не нужно.

Что важно для CPU:

- `kv-cache-memory-bytes` задан явно. Без него vLLM на CPU считает KV-кэш от памяти NUMA-ноды хоста,
  а не от лимита пода, и под ловит OOMKill.
- `VLLM_CPU_OMP_THREADS_BIND=nobind` и `OMP_NUM_THREADS = limits.cpu − 1` (считает entrypoint, лимит
  приходит через downward API). Режим по умолчанию `auto` вешает по потоку на каждое видимое ядро ноды, а
  CFS-квота пода только 4 CPU, это переподписка и троттлинг. Одно ядро остаётся API-серверу.
- requests = limits (Guaranteed QoS). На ноде со static CPU manager можно вернуть `auto`: под получит
  эксклюзивные ядра и привязка потоков к ним станет выгодной.

### Kubernetes

- Namespace `llm` с Pod Security Admission `restricted`: кластер сам отклонит под, который не rootless.
- Под: `runAsNonRoot`, uid/gid/fsGroup 10001, `seccompProfile: RuntimeDefault`, у обоих контейнеров
  `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, `readOnlyRootFilesystem: true`;
  отдельный ServiceAccount без токена API.
- `strategy: Recreate`: том RWO и ~10 GiB памяти на реплику, два пода рядом не нужны.
- startupProbe даёт до 15 минут на загрузку и `torch.compile`, потом readiness/liveness по `/health`.
- `/dev/shm` это `emptyDir` в памяти (дефолтных 64 MiB мало для обмена между процессами vLLM).
- Про NUMA: документация vLLM советует `SYS_NICE` и `seccomp: Unconfined` для привязки памяти к NUMA-нодам.
  Здесь это сознательно не сделано: на одной ноде minikube выигрыша нет, а профиль `restricted` важнее.

`tests/test_manifests.py` рендерит оба overlay и проверяет требования `restricted`, что секреты доходят только
до своих контейнеров и что ссылки на сгенерированные ConfigMap/Secret сходятся. Манифесты дополнительно
проверены по OpenAPI-схеме Kubernetes 1.34 (`kubernetes-validate --strict`).

## Нагрузочное тестирование

`loadtest/loadtest.py` (asyncio + aiohttp) держит заданное число одновременных клиентов и шлёт стриминговые
`/v1/chat/completions`. Каждый промпт начинается с уникального id, чтобы prefix cache не отвечал за соседа,
`ignore_eos` + `min_tokens` фиксируют длину ответа. Длина промпта калибруется по `usage.prompt_tokens` сервера.

Сценарии по умолчанию: **short** (~64 токена на входе, 128 на выходе; 1, 4 и 8 клиентов) и **long**
(~3000 токенов на входе, 128 на выходе; 1 и 4 клиента), на каждый уровень `4 × concurrency` запросов.

- **TTFT**: от отправки запроса до первого сгенерированного токена;
- **Latency**: от отправки до конца стрима; **TPOT** = (latency − TTFT) / (output_tokens − 1);
- **Throughput**: сгенерированные токены / время уровня (out tok/s), то же с токенами промпта (total tok/s), запросы/с.

```bash
pip install -r loadtest/requirements.txt
kubectl -n llm port-forward svc/vllm 8000:8000 &
python loadtest/loadtest.py --api-key "$VLLM_API_KEY" --update-readme README.md   # или: make loadtest
# свои сценарии: --scenario name:input_tokens:output_tokens:1,4,8 (можно несколько), --rounds 2 для быстрого прогона
```

Скрипт печатает таблицу, сохраняет `.md` и `.json` в `loadtest/results/`, а с `--update-readme` вписывает
таблицу сюда:

<!-- loadtest:begin -->

Предположения: Qwen2.5-1.5B-Instruct в bf16 (~3.1 GB весов), 8 vCPU x86 с AVX512 без AMX,
эффективная пропускная способность памяти ~25–35 GB/s.

| Сценарий | Клиенты | TTFT, с | TPOT, мс | Out tok/s (всего) |
|---|---|---|---|---|
| short (64 → 128) | 1 | 0.3–0.8 | 90–140 | 7–11 |
| short (64 → 128) | 4 | 0.5–1.5 | 110–180 | 22–36 |
| short (64 → 128) | 8 | 0.8–2.5 | 140–250 | 32–57 |
| long (3000 → 128) | 1 | 10–30 | 100–160 | 4–7 |
| long (3000 → 128) | 4 | 25–90 | 130–250 | 6–14 |

<!-- loadtest:end -->

Как читать результат: в short TTFT маленький и почти не растёт с concurrency, а out tok/s растёт за счёт
батчинга decode (веса читаются из памяти один раз на шаг для всего батча), TPOT при этом немного растёт.
В long TTFT определяется prefill: при 4 клиентах запросы стоят в очереди на prefill друг за другом
(чанками по `max-num-batched-tokens`), поэтому TTFT растёт почти линейно, а total tok/s почти целиком
состоит из входных токенов.

### Какие параметры менять для роста throughput

**А) Много запросов, мало токенов.** Узкое место это decode маленькими батчами плюс накладные расходы на
каждый запрос (HTTP, токенизация, детокенизация, планировщик).

- `--max-num-seqs` поднять (16 → 64…128). Decode на CPU упирается в пропускную способность памяти, веса
  читаются один раз на шаг на весь батч, поэтому чем шире батч, тем больше токенов за тот же проход.
  Цена: TPOT каждого запроса растёт. Это главный рычаг для сценария А.
- `--max-num-batched-tokens` держать умеренным (2048…4096): этого хватает, чтобы в один шаг уместить
  prefill нескольких коротких запросов вместе с decode остальных.
- KV-кэш (`--kv-cache-memory-bytes`) должен вмещать `max-num-seqs × (вход + выход)` токенов, иначе
  планировщик начнёт вытеснять запросы (preemption) и пересчитывать их. Для коротких запросов он
  небольшой, `--max-model-len` можно уменьшить до реально нужного.
- Фронтенд: на CPU он делит ядра с инференсом, поэтому под него резервируется ядро (здесь
  `OMP_NUM_THREADS = limits − 1`, в режиме `auto` это `VLLM_CPU_NUM_OF_RESERVED_CPU`). При очень большом
  потоке можно поднять `--api-server-count` и отключить access-лог uvicorn (`--disable-uvicorn-access-log`).
- `--enable-prefix-caching` (уже включён): общий системный промпт считается один раз.
- Масштабирование вширь: для маленькой модели несколько реплик с 4–8 потоками каждая обычно дают больше,
  чем одна реплика на все ядра, потому что маленькие GEMM плохо масштабируются по потокам. В k8s это
  `replicas` плюс RWX-том или StatefulSet с томом на реплику, либо `--data-parallel-size` внутри пода.
- Квантизация весов (на x86 vLLM CPU поддерживает AWQ, GPTQ, INT8 W8A8): меньше байт на вес, быстрее decode.

**Б) Мало запросов, огромный контекст.** Узкое место это prefill (вычисления растут линейно по GEMM и
квадратично по attention) и память под KV-кэш.

- `--max-num-batched-tokens` поднять (2048 → 8192…16384). Prefill идёт чанками такого размера, крупный чанк
  лучше загружает AVX512/AMX и требует меньше шагов, TTFT падает. По документации vLLM CPU этот параметр
  сильнее всего влияет именно на первый токен. Цена: на время большого чанка decode соседних запросов
  стоит (растёт TPOT), и нужно больше памяти под активации.
- Chunked prefill оставить включённым (в vLLM V1 он включён по умолчанию): он перемешивает prefill длинного
  запроса с decode остальных и не даёт одному запросу заблокировать всех.
- `--max-model-len` поднять до нужного контекста (Qwen2.5 держит 32k нативно, дальше нужен YaRN через
  `--hf-overrides`) и соразмерно поднять KV-кэш: для Qwen2.5-1.5B один токен занимает ~28 KiB в bf16, то есть
  32k токенов это ~0.9 GiB на каждую одновременную последовательность. Лимит памяти пода растёт вместе с ним.
  Нехватка KV-кэша при длинных запросах означает preemption и пересчёт prefill, а это самая дорогая ошибка.
- `--max-num-seqs` можно опустить до реальной параллельности: это не даёт throughput само по себе, но
  исключает вытеснения.
- `--enable-prefix-caching`: если длинные запросы делят префикс (один документ в RAG, история чата), повторный
  префикс не считается вообще. Для этого сценария выигрыш самый большой.
- Отдать prefill все физические ядра: он упирается в вычисления и масштабируется по ядрам. Потоки на
  физические ядра без hyper-threading, а на многосокетном сервере `--tensor-parallel-size` = числу NUMA-нод
  (так советует документация vLLM CPU), чтобы работала пропускная способность памяти обоих сокетов.
- bf16 на CPU с AMX (Xeon 4-го поколения и новее) ускоряет prefill в разы по сравнению с AVX2.

## Тесты

```bash
pip install -r requirements-dev.txt
make test        # скрипт скачивания, манифесты (нужен kustomize или kubectl), нагрузочный тест на фейковом сервере
make test-image  # логирование: внутри собранного образа на настоящем коде vLLM 0.32.0
```

## Windows

- Docker Desktop с WSL2-бэкендом, minikube с драйвером `docker`. Команды удобнее запускать в WSL2 (Ubuntu):
  там есть bash, make, `eval $(minikube docker-env)`. Из PowerShell образы собираются после
  `& minikube -p minikube docker-env --shell powershell | Invoke-Expression`.
- WSL2 по умолчанию получает только половину RAM хоста. Для профиля 14g в `%UserProfile%\.wslconfig`:
  `[wsl2]` `memory=16GB` `processors=8`, затем `wsl --shutdown`. Если памяти меньше, используйте
  `minikube start --cpus=4 --memory=8g` и `kubectl apply -k k8s/overlays/minikube-small`.
- `.gitattributes` закрепляет LF для скриптов: с CRLF `entrypoint.sh` в Linux-контейнере не запустится.

## Если что-то пошло не так

- Под в `OOMKilled`: поднять `limits.memory` в Deployment или уменьшить `kv-cache-memory-bytes`, либо взять
  overlay `minikube-small`.
- startupProbe не дождалась: на медленном CPU поднять `failureThreshold` у startupProbe и
  `VLLM_ENGINE_READY_TIMEOUT_S` в `vllm.env`.
- Docker Hub недоступен: базовый образ меняется аргументом сборки,
  `--build-arg PYTHON_IMAGE=<зеркало>/python:3.12-slim-trixie` (и `python:3.12-alpine` для загрузчика).
- `pip check` упал на сборке: значит, метаданные опубликованного wheel отличаются от lock-файла;
  `./scripts/lock-deps.sh` пересоберёт lock.
