# Approved anonymized narrative benchmark

This directory is the local/private entrypoint for Layer D evaluation data. Raw family audio, unrestricted transcripts, names, contact details, and other identifying material must not be committed.

The production release profile expects `approved_real.json` to be supplied through an approved private-data workflow and the matching manifest entry to be enabled with:

- `layer: anonymized_real`;
- `approved_anonymized: true`;
- `narrator_count >= 3`;
- `required_for_production: true`.

The file uses the same `BenchmarkDataset` JSON contract as public fixtures. Every case must contain a hand-verified anonymized transcript, fixed candidate extraction, and expected people/relationship graph. Replacing names with stable pseudonyms is required; simply deleting surnames is not sufficient anonymization.

The public pull-request gate does not pretend this dataset exists. The production gate fails when it is absent, disabled, unapproved, or has fewer than three independent narrators.
