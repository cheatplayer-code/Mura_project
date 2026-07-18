from __future__ import annotations

from mura.deepseek.prompts import EXTRACTION_REPAIR_SYSTEM_PROMPT, EXTRACTOR_SYSTEM_PROMPT


ANCHOR_CONSTRAINTS = """
Anchor contract rules:
1. anchor_contract.allowed_segment_ids is the complete set of segment IDs you may cite.
2. mention_anchors are candidate identity surfaces, not permission to invent a person or merge two
   people. A known_person_id identifies an existing archive candidate, not a persistent person ID
   for output claims.
3. Every returned PersonMention must still be supported by literal transcript evidence. When a
   mention_anchor matches the intended person, keep its surface and segment grounding consistent.
4. lexical_annotations are deterministic hints only. They may guide candidate generation, but they
   are not evidence and must never be copied into evidence_spans as if the model observed them.
5. A relationship endpoint must reference a PersonMention returned in the same JSON object. Never
   reference known_person_id, anchor_id, or an unreturned person as an endpoint.
6. If a claimed endpoint is absent from both the transcript and mention_anchors, omit the claim and
   add an unresolved question instead of inventing the endpoint.
7. Do not expand the anchor world during repair. The repair response must use the exact same
   anchor_contract and allowed segment IDs supplied in the repair payload.
8. Anchor presence does not prove relationship type, direction, alias identity, or coreference.
   Those still require exact transcript evidence and the normal evidence-class rules.
""".strip()


ANCHOR_CONSTRAINED_EXTRACTOR_SYSTEM_PROMPT = (
    f"{EXTRACTOR_SYSTEM_PROMPT}\n\n{ANCHOR_CONSTRAINTS}"
)

ANCHOR_CONSTRAINED_EXTRACTION_REPAIR_SYSTEM_PROMPT = (
    f"{EXTRACTION_REPAIR_SYSTEM_PROMPT}\n\n{ANCHOR_CONSTRAINTS}"
)
