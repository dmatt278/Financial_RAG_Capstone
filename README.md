# Financial RAG Capstone

An experimental retrieval-augmented generation pipeline for financial question answering on [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA). The project builds a persistent Chroma index, tunes retrieval parameters without generation calls, selects a configuration on a development sample, and compares the frozen configuration with several baselines on a held-out test sample.

This repository is primarily an experiment API and evaluation pipeline. It is not currently a general-purpose chat application: retrieval is evaluated within the question's known source document, and the React directory is an unconnected Vite starter.

## Implemented workflow

```text
DocFinQA documents ──> 9-way Chroma index ──> Stage 1 retrieval sweep
                                                       │
FinQA gold evidence ──> evidence/chunk alignment       ├─> top 3 configurations
                                                       │
                                                       v
                                           Stage 2 dev generation
                                                       │
                                                       └─> frozen winner
                                                              │
                                                              v
                                            Stage 3 held-out baselines
                                                              │
                                                              v
                                              PostgreSQL + final report
```

The core experiment uses a fixed random seed (`42`) and uniform sampling without replacement. Question IDs are saved with each logged run so an interrupted run can resume with exactly the same sample.

| Stage | Dataset | Default work | Selection or comparison |
|---|---|---:|---|
| Index setup | train, dev, test documents | 3 chunking strategies × 3 chunk sizes | Stores all nine representations in Chroma |
| Stage 1: `/run-chunk-rag` | 50 train questions | 324 retrieval configurations per question; no OpenAI calls | Saves the top three retrieval configurations |
| Stage 2: `/run-full-rag` | 100 dev questions | Top 3 × 100 = 300 generated answers before retries | Saves one model-specific winner |
| Stage 3: `/get-baselines` | 100 test questions | 4 methods × 100 = 400 generated answers before retries | Compares the frozen winner with three baselines |
| Optional math experiment | 50 train+dev questions | Direct LLM vs. LLM-planned deterministic calculation | Separate exploratory analysis; not part of Stage 1–3 selection |

Only the exact default protocols can save or replace the authoritative shortlist, winner, and statistical analyses. Smaller samples and partial runs are useful for diagnostics, but they deliberately cannot overwrite those artifacts.

## Retrieval and evaluation design

### Indexing

- Source dataset: `kensho/DocFinQA` from Hugging Face.
- Documents are deduplicated across the train, dev, and test questions.
- Chunking strategies: `fixed`, `sentence`, and `section`.
- Chunk sizes: 256, 512, and 1024 tokens, without overlap.
- Embeddings: `jinaai/jina-embeddings-v2-small-en`, normalized to unit length.
- Vector store: persistent Chroma collection `docfinqa_chunks`.
- Chroma writes are batched in groups of 1,000 chunks.

### Retrieval

Every query is filtered to the example's known `document_id`, chunking strategy, and chunk size. The experiment therefore measures **within-document chunk retrieval**, not corpus-wide source-document discovery.

The Stage 1 grid contains:

- retrieval method: BM25 keyword, Chroma semantic, or hybrid;
- chunking strategy: fixed, sentence, or section;
- chunk size: 256, 512, or 1024;
- `top_k`: 3, 5, or 10;
- reranking: off, or on with a candidate pool of 10, 20, or 40.

This gives `3 × 3 × 3 × 3 × 4 = 324` configurations. Hybrid retrieval uses reciprocal-rank fusion. Optional reranking uses `jinaai/jina-reranker-v1-tiny-en` at the revision pinned in `retriever.py`.

Stage 1 ranks configurations with the predeclared score:

```text
0.50 × all-evidence-hit@k
+ 0.30 × evidence-recall@k
+ 0.10 × reciprocal rank
+ 0.10 × precision@k
```

Ties prefer the individual retrieval metrics in that order, followed by the simpler configuration: no reranker, a smaller reranker pool, smaller `top_k`, and a smaller chunk size.

### Gold evidence and answer scoring

The original FinQA train/dev/test files provide supporting facts. The code matches those facts to DocFinQA questions and aligns each fact with the best chunk using four-gram overlap. A retrieved chunk is relevant when it is one of the best-aligned chunks for a gold supporting fact. This is a reproducible heuristic, not human relevance annotation.

Stage 2 selects the winner by:

1. DocFinQA answer accuracy;
2. within-1% accuracy;
3. retrieval metrics;
4. deterministic simplicity tie-breakers.

Numeric answer correctness accepts values that round to the precision displayed by the gold answer. The pipeline also records parse success, exact normalized match, absolute and relative error, and within-1% accuracy.

Statistical reporting includes Wilson confidence intervals, Cochran's Q, exact paired McNemar tests, and Holm-adjusted p-values. Stage 2 statistics are exploratory diagnostics and do not choose the winner. Stage 3 statistics compare the frozen optimized system with each baseline on held-out questions.

## Repository layout

```text
backend/
  app/
    data/data_loader.py       DocFinQA and FinQA loading and sampling
    rag/chunker.py            Fixed, sentence, and section chunking
    rag/embedder.py           Embedding model and CPU/CUDA selection
    rag/retriever.py          BM25, semantic, hybrid, and reranking
    rag/generator.py          OpenAI answer generation
    rag/pipeline.py           Staged experiment orchestration
    rag/parameter_selection.py
    rag/math_agent.py         Optional experimental calculation pipeline
    rag/vector_store.py       Persistent Chroma indexing
    evaluation.py             Retrieval and answer metrics
    results_logger.py         PostgreSQL schema, results, and checkpoints
    statistical_analysis.py  Paired statistical tests
    main.py                   FastAPI routes
  tests/                      Unit and mocked integration tests
frontend/                     React/Vite starter; not connected to the API
notebooks/                    Exploratory notebooks
data/                         Local data and Chroma storage; gitignored
docker-compose.yml            Backend plus PostgreSQL for local CPU use
```

## Requirements

- Python 3.11 is the tested container version.
- Internet access is needed to download DocFinQA, FinQA, and Hugging Face models, and to call OpenAI for Stages 2 and 3.
- PostgreSQL is required for logged experiments, checkpoints, saved parameters, and reports.
- An OpenAI API key is required for Stage 2, Stage 3, and the optional math experiment. It is not required for indexing or Stage 1.
- A CUDA GPU is optional. The code uses one CUDA device when available and does not distribute work across multiple GPUs.

## Quick start with Docker Compose

Docker Compose starts the backend at `http://127.0.0.1:8000` and PostgreSQL 16. The checked-in Compose configuration is a CPU-oriented local setup and does not request an NVIDIA runtime.

Create a private environment file:

```bash
cp .env.example .env
```

On PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Add an API key to `.env` before running generation stages:

```dotenv
OPENAI_API_KEY=your_key_here
```

Then start the services:

```bash
docker compose up --build -d
docker compose logs -f backend
```

Check the API and its interactive documentation:

```bash
curl http://127.0.0.1:8000/health
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

The Compose services use named volumes for PostgreSQL, Chroma, DocFinQA, FinQA, and the Hugging Face cache. `docker compose down` preserves them. **`docker compose down -v` deletes them.**

## Native Python or RunPod setup

Create and install a Python environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

PowerShell equivalent:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

For native execution, set a reachable PostgreSQL URL. The default hostname `db` exists only inside Docker Compose. An example private environment file is:

```dotenv
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_CONCURRENCY=4
OPENAI_MAX_RETRIES=2
DOCFINQA_DATA_DIR=/absolute/path/to/data/docfinqa
FINQA_DATA_DIR=/absolute/path/to/data/finqa
CHROMA_PERSIST_DIR=/absolute/path/to/data/chroma
RAG_DEVICE=auto
EMBEDDING_BATCH_SIZE=1
RERANKER_BATCH_SIZE=16
```

The application does not load `.env` by itself. Either export/source the variables first, or have Uvicorn load the file:

```bash
python -m uvicorn app.main:app \
  --app-dir backend \
  --env-file .env \
  --host 0.0.0.0 \
  --port 8000
```

Use one Uvicorn worker. Long experiment routes share one process-local lock; a second long request receives HTTP 409.

### RunPod notes

A RunPod PyTorch image can provide CUDA without running Docker inside the pod. Recommended practices:

- mount the persistent network volume at `/workspace`;
- keep the durable Chroma source at `/workspace/data/chroma`;
- persist the project, datasets, and Hugging Face cache under `/workspace`;
- use an external PostgreSQL service such as Neon through `DATABASE_URL`;
- set `RAG_DEVICE=cuda` to fail early if CUDA is unavailable;
- start both Uvicorn and long `curl` calls inside separate `tmux` sessions.

Network-volume Chroma survives pod replacement but can be slow. A copy placed on container-local disk may be faster, but it is ephemeral. Stop Uvicorn before copying a complete Chroma directory, point `CHROMA_PERSIST_DIR` at the copy, and restart Uvicorn because Chroma clients are cached in the process.

Verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Verify the Chroma collection after loading the environment:

```bash
cd backend
python -c "from app.rag.vector_store import get_collection; print(get_collection().count())"
```

## Build or reuse the datasets and index

First download and validate the original FinQA evidence files:

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/load-finqa' \
  -o load-finqa-result.json
```

For a brand-new Chroma index only, build all nine chunk configurations:

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/load-docfinqa?reset=true' \
  -o load-docfinqa-result.json
```

> **Warning:** `reset=true` deletes and recreates the Chroma collection before indexing. Do not use it when a complete reusable index already exists. Indexing produces millions of embeddings, can require substantial storage, and can take many hours. This route does not use the PostgreSQL checkpoint system.

With `reset=false`, IDs are deterministic and Chroma upserts them, but a full repeat still re-embeds and overwrites the chunks and is therefore expensive.

## Run the experiment

Run the core stages in order. Leave `return_results=false` for full runs; the detailed rows are already written to PostgreSQL, while returning them can create very large JSON responses.

### Stage 1: retrieval-only train sweep

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/run-chunk-rag?sample_size=50&sample_seed=42&log_results=true&return_results=false' \
  -o run-chunk-rag-result.json
```

The authoritative run creates 16,200 result rows and saves the top three configurations. It does not call OpenAI.

### Stage 2: dev generation and winner selection

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/run-full-rag?sample_size=100&sample_seed=42&log_results=true&return_results=false' \
  -o run-full-rag-result.json
```

This route requires the authoritative Stage 1 shortlist in the same PostgreSQL database. It evaluates all three finalists on the same 100 dev questions and saves a winner for the current `OPENAI_MODEL`.

### Stage 3: held-out baseline comparison

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/get-baselines?sample_size=100&sample_seed=42' \
  -o get-baselines-result.json
```

This route requires the saved Stage 2 winner for the current model and evaluates four paired methods:

1. question-only OpenAI answer (`no_context`);
2. prompt-budgeted full-document answer (`full_document`);
3. naive semantic RAG using fixed chunks, size 512, `top_k=3`, and no reranker;
4. optimized RAG using the frozen dev winner.

The full-document input uses the same assumed 8,192-token prompt budget as the other generation paths and may be truncated for long reports.

### Optional math-agent experiment

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/run-full-rag-math-agent?retrieval_method=hybrid&strategy=section&chunk_size=512&top_k=5&return_results=false' \
  -o math-agent-result.json
```

This is separate from the core Stage 1–3 experiment. It compares a direct LLM answer with an LLM-generated calculation program executed by a deterministic tool on 50 train+dev questions. The endpoint does not consume the saved Stage 2 winner and always disables reranking. A program-repair attempt can make its OpenAI request count exceed two calls per question.

### View final results

```bash
curl --fail-with-body -sS \
  'http://127.0.0.1:8000/final-results?model=gpt-4o-mini' \
  -o final-results.json

python -m json.tool final-results.json
```

`/final-results` reads PostgreSQL and reports the latest completed authoritative runs, the saved shortlist and winner, statistical analyses, baseline summaries, and any separate exploratory math result.

## Progress and resume

Logged experiment routes create a `run_id`, print it in the API logs, and store a durable manifest in PostgreSQL. Check one run with:

```bash
curl --fail-with-body -sS \
  'http://127.0.0.1:8000/experiment-status/RUN_ID' \
  | python -m json.tool
```

The status includes saved and expected row counts, question progress, timestamps, parameters, and the sampled question IDs. It does not calculate an ETA.

If a logged run fails while its status remains `running`, call the same endpoint with the same model, parameters, sample size, and seed, adding its ID. For example:

```bash
curl --fail-with-body -sS -X POST \
  'http://127.0.0.1:8000/run-full-rag?sample_size=100&sample_seed=42&log_results=true&return_results=false&resume_run_id=RUN_ID' \
  -o run-full-rag-resumed.json
```

Resume reloads the saved question IDs and skips result keys already stored for that run. A completed run cannot be extended. Runs created with `log_results=false` cannot be resumed.

Because the POST routes are synchronous, use `tmux` on a remote machine:

```bash
tmux new-session -s financial-rag-api
# start Uvicorn, then detach with Ctrl-b d

tmux new-session -s stage1
# run the Stage 1 curl command, then detach with Ctrl-b d
```

Reattach with `tmux attach-session -t financial-rag-api` or `tmux attach-session -t stage1`.

## API reference

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Basic backend message |
| GET | `/health` | Lightweight process check only; it does not test PostgreSQL, Chroma, OpenAI, data, or GPU |
| POST | `/load-finqa` | Download and validate original FinQA evidence files |
| POST | `/load-docfinqa` | Chunk, embed, and upsert unique DocFinQA documents into Chroma |
| POST | `/run-chunk-rag` | Stage 1 retrieval-only parameter sweep |
| POST | `/run-full-rag` | Stage 2 generation over the saved shortlist |
| POST | `/get-baselines` | Stage 3 frozen-winner baseline comparison |
| POST | `/run-full-rag-math-agent` | Optional direct-LLM versus math-tool experiment |
| GET | `/experiment-status/{run_id}` | Read a persisted run checkpoint |
| GET | `/final-results` | Read saved summaries and statistical analyses |

FastAPI documents every query parameter and response shape at `/docs`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@db:5432/financial_rag` | PostgreSQL connection; the default works only in Compose |
| `OPENAI_API_KEY` | none | Required for generation stages |
| `OPENAI_MODEL` | `gpt-4o-mini` | Generation model and part of the saved-winner identity |
| `OPENAI_CONCURRENCY` | `4` | Maximum concurrent generation tasks |
| `OPENAI_MAX_RETRIES` | `2` | OpenAI SDK retry count |
| `DOCFINQA_DATA_DIR` | repository `data/docfinqa` | DocFinQA local data directory |
| `FINQA_DATA_DIR` | repository `data/finqa` | Original FinQA evidence directory |
| `CHROMA_PERSIST_DIR` | relative `data/chroma` | Persistent Chroma directory; an absolute path is recommended |
| `RAG_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `EMBEDDING_BATCH_SIZE` | `1` | Embedding inference batch size |
| `RERANKER_BATCH_SIZE` | `16` | Cross-encoder inference batch size |
| `HF_HOME` | Hugging Face default | Optional persistent Hugging Face cache location |

The `OPTIMIZED_*` values in `.env.example` are legacy/fallback configuration values. The authoritative Stage 3 pipeline does not treat them as a substitute for the Stage 2 winner stored in PostgreSQL.

## PostgreSQL and storage

The schema is created automatically and includes:

- `rag_results` for per-question outcomes;
- `rag_experiment_runs` for manifests and checkpoints;
- `rag_retrieval_shortlists` for the Stage 1 finalists;
- `rag_best_parameters` for the Stage 2 winner;
- `rag_statistical_analyses` for paired analyses.

Chroma and PostgreSQL are independent:

- Chroma stores documents, chunks, and embeddings at `CHROMA_PERSIST_DIR`.
- PostgreSQL stores experiment results and the Stage 1→2→3 handoff artifacts.
- Docker named volumes survive a normal Compose shutdown.
- A RunPod network volume survives pod replacement; container-local files may not.
- Neon PostgreSQL persists independently of the pod.

Stage 2 requires the Stage 1 shortlist in the same PostgreSQL state. Stage 3 requires the Stage 2 winner in the same PostgreSQL state and for the same `OPENAI_MODEL`. Merely knowing the parameter values does not satisfy those database guards.

## Tests

Run the backend test suite from `backend`:

```bash
cd backend
python -m unittest discover -s tests -v
```

The current suite contains 108 unit and mocked integration tests. These tests validate pipeline logic but do not replace a live integration test against Hugging Face, Chroma, PostgreSQL, and OpenAI.

The frontend starter can be checked separately:

```bash
cd frontend
npm install
npm run lint
npm run build
```

## Current limitations

- Retrieval assumes the correct source document is already known.
- FinQA-to-DocFinQA evidence alignment is heuristic four-gram matching.
- Long experiment endpoints are blocking HTTP requests and allow only one long task per API process.
- The Docker Compose file does not configure GPU access; CUDA requires an environment with CUDA-enabled PyTorch and GPU access.
- Embedding and reranking use one selected device, not multiple GPUs.
- The prompt-budgeted full-document baseline can truncate long reports.
- The optional math-agent endpoint is exploratory, forces reranking off, and does not use the saved Stage 2 winner.
- The React/Vite frontend is a starter page with no API integration. There is no interactive `/ask` endpoint.
- The API has no authentication or production hardening.
