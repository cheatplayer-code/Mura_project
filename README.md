# Mura (Мұра)

Mura turns informal Kazakh, Russian, and mixed-language family voice stories into a source-linked family archive: readable transcripts, people, relationships, events, stories, corrections, and unresolved questions.

## Current pipeline

```text
phone audio
  -> Kaggle GPU worker (FFmpeg -> Silero VAD -> smart chunks -> GigaAM large_ctc)
  -> immutable raw transcript with timestamps
  -> DeepSeek faithful cleaner
  -> DeepSeek family-memory extractor
  -> Pydantic and cross-reference validation
  -> mentions, review queue, and family graph
```

The GPU worker and the core API are deliberately separate. Kaggle provides temporary T4 compute; the core API owns DeepSeek calls and future persistence. No model fine-tuning is required for the MVP.

## Repository layout

```text
apps/api/                 Core FastAPI service
services/kaggle_asr/      Kaggle T4 ASR worker and tunnel bootstrap
src/mura/domain/          Stable Pydantic contracts
src/mura/deepseek/        HTTP client, prompts, cleaner, extractor
tests/                    Unit and contract tests
docs/                     Architecture and decisions
notebooks/                Kaggle launch instructions
```

## Local core API

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn apps.api.main:app --reload --port 8001
```

The core API does not need a GPU. Set `DEEPSEEK_API_KEY` before calling the processing endpoint.

## Kaggle GPU worker

1. Select **GPU T4** and enable Internet.
2. Add Kaggle secrets: `KAGGLE_ASR_API_KEY` and optionally `HF_TOKEN`.
3. Clone this repository.
4. Run the commands in [`notebooks/KAGGLE.md`](notebooks/KAGGLE.md).

The worker exposes:

- `GET /health`
- `GET /model-info`
- `POST /v1/transcribe` with `Authorization: Bearer <KAGGLE_ASR_API_KEY>`

Quick Tunnel URLs are temporary and live only while the Kaggle session and `cloudflared` process remain active.

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Safety rules

- Raw ASR text is never overwritten.
- Every extracted object must cite real source segment IDs.
- New stories default to `private`.
- LLM confidence is not treated as verification.
- Ambiguous entity matches go to review instead of being merged automatically.
- API keys and model weights are never committed.
