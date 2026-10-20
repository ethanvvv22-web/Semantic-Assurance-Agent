# Semantic Assurance Agent

Verify that AI-agent transformations of operator intent into network decisions
and executable configurations respect the **original constraints and
guardrails** — using **semantic equivalence chains** rather than a direct
input/output comparison.

Reference implementation for the Ericsson autonomous-networks thesis
(Req ID 791334). Runnable MVP with a baseline comparison and measured results.

> `intent (NL) -> structured decision -> executable action (API call)`
> e.g. *"close my account"* -> *"user-terminated subscription"* -> `POST /billing/v1/.../cancel`

## The idea

A direct comparison asks only "does the output look like the input?". That
misses breaches where the action still references the right entities and reads
like the intent but **exceeds a bound, drops a time window, adds an unrequested
side effect, or disables an emergency bearer**.

The Semantic Assurance Agent builds an equivalence chain and checks each link:

```
intent_text  ──►  canonical_intent  ──►  verbalized_action
     └──────────────── end-to-end semantic similarity ─────────────┘
                        ▲ reconstruct-the-intent link ▲
```

It combines the chain's **semantic-preservation score** with a declarative
**constraint/guardrail catalog**, then grades the transformation as
**acceptable / review / breach** — making explicit where the line between an
acceptable paraphrase-level deviation and a real breach sits.

## JD keyword → where it lives

| JD keyword | Module |
|---|---|
| intent → decision → executable config transformations | [saa/pipeline.py](saa/pipeline.py) |
| guardrails & constraints verified and enforced | [saa/constraints.py](saa/constraints.py) + [saa/constraints.yaml](saa/constraints.yaml) |
| **semantic equivalence chains** | [saa/chain.py](saa/chain.py), [saa/embedding.py](saa/embedding.py) |
| **Semantic Assurance Agent** checking agentic transformations | [saa/assurance.py](saa/assurance.py) |
| recognize deviations beyond direct input/output comparison | [saa/baseline.py](saa/baseline.py) vs [saa/assurance.py](saa/assurance.py) |
| define an **"acceptable" guardrail deviation** | [saa/deviation.py](saa/deviation.py) + thresholds in [saa/config.py](saa/config.py) |
| experimental setup + **baseline comparison** | [eval/run_eval.py](eval/run_eval.py), [data/dataset.py](data/dataset.py) |
| AI concepts | local sentence-transformers embeddings; optional LLM back-translation ([saa/llm.py](saa/llm.py)) |

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; or .venv/bin/activate
pip install -r requirements.txt

python -m demo.run_demo                 # 5 transformations, chain + verdict + baseline
python -m eval.run_eval                 # baseline vs SAA: P/R/F1, confusion, extra catches
python -m unittest discover -s tests    # 15 tests (offline, TF-IDF backend)
```

First run downloads a ~80 MB embedding model (`all-MiniLM-L6-v2`); afterwards it
is fully offline. If the model can't be fetched, the engine falls back to TF-IDF
automatically so everything still runs.

## Measured result (all-MiniLM-L6-v2, 41-record dataset)

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Direct input/output comparison (baseline) | 1.00 | 0.29 | 0.44 |
| **Semantic Assurance Agent** | 0.85 | **1.00** | **0.92** |

The chain catches **20 extra true breaches** the direct comparison misses
(bound, temporal, over-reach, emergency). The remaining SAA false positives are
wordy-but-benign paraphrases landing in the **review** band — exactly the
"acceptable deviation" boundary the thesis is meant to explore.

## Roadmap to thesis depth

Larger / real intent corpora · LLM-generated equivalence chains & back-translation
([saa/llm.py](saa/llm.py)) · learned (not hand-set) acceptable-deviation thresholds
with ROC analysis · multi-hop chains over real O-RAN/3GPP intent models ·
inter-annotator ground truth · ablations (constraints-only vs chain-only vs combined).
