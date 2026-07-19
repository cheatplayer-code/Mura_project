# Local testing for Mura Core RC1

The fastest verification path does not require Docker, Kaggle, GigaAM, DeepSeek, or API keys.

## 1. Install the project

```bash
python -m pip install -e ".[dev]"
```

## 2. Run the one-command release smoke test

```bash
mura-release-smoke
```

The command creates an isolated temporary SQLite database and verifies:

- recording and job persistence;
- atomic result completion;
- runtime performance and cost-unit budgets;
- privacy-safe processing traces;
- deterministic family replay in a shadow database;
- release rollback and restoration;
- retention preview and confirmed operational cleanup.

A successful run exits with code `0` and prints:

```json
{
  "release_id": "mura-core-v1.0.0-rc1",
  "passed": true,
  "checks": {
    "job_completed": true,
    "budget_passed": true,
    "trace_available": true,
    "replay_passed": true,
    "rollback_requires_restart": true,
    "release_restored": true,
    "retention_preview_found_data": true,
    "retention_deleted_data": true
  }
}
```

Save the complete report and database when diagnosing a problem:

```bash
mura-release-smoke \
  --database-path .mura/smoke/mura-smoke.db \
  --json-output .mura/smoke/report.json
```

## 3. Run the public ML release gates

```bash
mura-evaluate-core \
  --manifest benchmarks/release_manifest.json \
  --release-gates benchmarks/release_gates.json \
  --json-output benchmark-report.json \
  --markdown-output benchmark-report.md
```

## 4. Run the full test suite

```bash
pytest -ra
```

## 5. Optional PostgreSQL verification

GitHub Quality CI automatically starts PostgreSQL, applies every Alembic migration through `20260719_0006`, and runs the PostgreSQL atomicity test.

For a local database:

```bash
export DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/mura'
alembic upgrade head
```

The smoke command intentionally uses SQLite so that basic verification remains one command. PostgreSQL remains the product database and is verified in CI.

## Interpreting replay

Deterministic replay does not call GigaAM or DeepSeek. It takes stored immutable pipeline results, reruns current schema/evidence validation, rebuilds the family archive in an isolated shadow database, and returns a canonical snapshot hash.

Human conflict decisions are not replayed into that shadow projection. The report states this explicitly. Live LLM replay would measure provider variability rather than deterministic reproducibility and is therefore outside RC1.

## Interpreting rollback

Rollback updates the desired release and writes an audit decision. It does not hot-swap Python code inside a running process. When the desired release differs from the current runtime, the API returns `restart_required: true`; the matching historical runtime must then be deployed.
