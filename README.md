# Mura (Мұра)

Mura turns informal Kazakh, Russian, and mixed-language family voice stories into a source-linked family archive: readable transcripts, people, relationships, events, stories, corrections, and unresolved questions.

## Current pipeline

```text
mobile application
  -> Core FastAPI upload endpoint
  -> PostgreSQL processing job
  -> Kaggle GPU worker (FFmpeg -> Silero VAD -> smart chunks -> GigaAM large_ctc)
  -> immutable raw transcript with timestamps
  -> DeepSeek faithful cleaner
  -> DeepSeek family-memory extractor
  -> Pydantic, reference, and semantic validation
  -> quarantine and mention resolution
  -> PostgreSQL pipeline result
  -> polling and review endpoints for the application
```

The GPU worker and the core API are deliberately separate. Kaggle provides temporary T4 compute. The core API owns authentication, durable jobs, DeepSeek calls, PostgreSQL persistence, and the application-facing contract. No model fine-tuning is required for the MVP.

## Repository layout

```text
apps/api/                 Core FastAPI service
services/kaggle_asr/      Kaggle T4 ASR worker and tunnel bootstrap
src/mura/asr/             Remote ASR worker client
src/mura/deepseek/        HTTP client, prompts, cleaner, extractor
src/mura/orchestration/   Audio storage and background job worker
src/mura/storage/         SQLAlchemy models and repository
src/mura/domain/          Stable Pydantic contracts
migrations/               Alembic PostgreSQL migrations
tests/                    Unit and contract tests
docs/                     Architecture and decisions
notebooks/                Kaggle launch instructions
```

## Local Core API with PostgreSQL

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
uvicorn apps.api.main:app --reload --port 8001
```

For hackathon development `DATABASE_AUTO_CREATE=true` can create the initial schema automatically. A deployed environment should run `alembic upgrade head` and set `DATABASE_AUTO_CREATE=false`.

The Core API does not need a GPU. It requires PostgreSQL, `DEEPSEEK_API_KEY`, `CORE_API_KEY`, `KAGGLE_ASR_API_KEY`, and `WORKER_REGISTRATION_TOKEN`.

## Application API contract

All application endpoints require:

```http
Authorization: Bearer <CORE_API_KEY>
```

Upload a recording:

```http
POST /v1/recordings
Content-Type: multipart/form-data

file=<audio>
family_id=family_001
speaker_id=person_kulash
speaker_name=Күләш
```

The API responds immediately with HTTP `202`:

```json
{
  "recording_id": "rec_...",
  "job_id": "job_...",
  "status": "queued"
}
```

Poll processing state:

```http
GET /v1/jobs/{job_id}
```

Possible statuses are `queued`, `transcribing`, `cleaning`, `extracting`, `resolving`, `completed`, and `failed`. When Kaggle is offline, the job stays queued with stage `waiting_for_asr` and is retried after the worker registers again.

Read the completed source-linked result:

```http
GET /v1/recordings/{recording_id}
```

Read only items that require a human decision:

```http
GET /v1/recordings/{recording_id}/review-items
```

FastAPI publishes the exact interactive contract at `/docs` and the OpenAPI document at `/openapi.json`.

## Kaggle GPU worker

1. Select **GPU T4** and enable Internet.
2. Add Kaggle secrets: `KAGGLE_ASR_API_KEY` and optionally `HF_TOKEN`.
3. Set `CORE_BACKEND_URL` and `WORKER_REGISTRATION_TOKEN` to register the current tunnel URL.
4. Clone this repository.
5. Run the commands in [`notebooks/KAGGLE.md`](notebooks/KAGGLE.md).

The worker exposes:

- `GET /health`
- `GET /model-info`
- `POST /v1/transcribe` with `Authorization: Bearer <KAGGLE_ASR_API_KEY>`

Quick Tunnel URLs are temporary and live only while the Kaggle session and `cloudflared` process remain active. The latest URL is persisted in PostgreSQL by the Core API.

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src apps services
```

## Safety rules

- Raw ASR text is never overwritten.
- Every extracted object must cite real source segment IDs.
- New stories default to `private`.
- LLM confidence is not treated as verification.
- Invalid isolated model objects are quarantined instead of poisoning the result.
- Ambiguous entity matches go to review instead of being merged automatically.
- Application, worker-registration, Kaggle ASR, and DeepSeek credentials are separate.
- API keys and model weights are never committed.
