# Mura architecture

## Goal

Mura preserves family memory without pretending that model output is ground truth. Audio, raw ASR, readable transcript, extracted claims, and the materialized family graph are separate layers with separate responsibilities.

## Components

```text
Mobile/web client
  -> Core API (ordinary CPU hosting)
       -> Kaggle ASR worker through HTTPS tunnel
            -> FFmpeg -> Silero VAD -> smart chunks -> GigaAM large_ctc
       <- source-linked raw transcript
       -> DeepSeek cleaner
       -> DeepSeek extractor
       -> Pydantic and graph-reference validators
       -> mention resolver / review queue
       -> future database and family graph
```

### Kaggle ASR worker

The worker accepts an audio upload and returns an immutable `TranscriptEnvelope`. It never calls DeepSeek and never mutates family data.

Security and stability rules:

- bearer authentication;
- one ASR job at a time;
- upload size and duration limits;
- temporary files are deleted after each request;
- no user-provided filesystem paths;
- Swagger UI is disabled on the public worker;
- a Quick Tunnel URL is treated as temporary.

### Smart chunking

1. Convert audio to 16 kHz mono PCM WAV.
2. Silero VAD identifies speech regions.
3. Nearby regions are merged while the result remains below 22 seconds.
4. Add 450 ms context at both boundaries, keeping the final chunk below 25 seconds.
5. Transcribe every chunk with GigaAM `large_ctc`.
6. Remove a textual boundary duplicate only when the underlying audio ranges overlap.

The last condition protects genuine human repetition.

### Core processing

The cleaner may add punctuation and mark uncertainty. It may not translate, invent, or silently alter names and numbers.

Every extracted person, relationship, event, description, story, and unresolved question must reference existing transcript segments. All generated objects start as `unreviewed`; every new story is forced to `private` by the application schema.

### Mention resolution

A mention is an occurrence in one recording. A person is a durable graph node. The first resolver only auto-resolves an exact canonical-name or explicit-alias match with compatible relation context. Ambiguous matches become `needs_review`; unmatched mentions become `new_person` candidates.

## Data invariants

- `RawSegment` text is never overwritten.
- Segment IDs are unique inside a recording.
- Cleaned output covers exactly the same segment IDs as raw output.
- All extraction references resolve.
- A relationship cannot connect a mention to itself.
- LLM confidence is not equivalent to user confirmation.
- Publication is always an explicit user action.

## Known limitation

Kaggle is a temporary GPU worker. A Quick Tunnel works only while the notebook session, Uvicorn, and `cloudflared` remain alive. The core API treats the worker URL as a replaceable runtime registration, not permanent infrastructure.
