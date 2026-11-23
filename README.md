# Ontology Reasoning vs. Local LLMs — Benchmark and Evaluation Framework

This repository contains the code, benchmark, and raw results for an empirical
study of whether local, open-weight LLMs can reproduce the answers obtained by
querying OWL ontologies with SPARQL, across ontologies of very different sizes
(44 KB to 130 MB).

Six models served locally through [Ollama](https://ollama.com)
(`gpt-oss:20b`, `qwen3.8:27b`, `qwen2.5:7b`, `llama3.1:8b`,
`granite3.3:8b`, `granite4.2:8b`) are evaluated under four prompting modes:

| Mode             | What the model receives                                              |
|------------------|----------------------------------------------------------------------|
| `Q`              | The question only.                                                   |
| `Q+Domain`       | The question plus a domain and answer-format constraint.             |
| `Q+Onto+Domain`  | The question plus the **whole ontology** pasted in the system prompt (plus the reference SPARQL query). |
| `RAG_a`          | The question plus the 35 ontology triples most similar to it, retrieved from a vector index. |

## Repository layout

```
main.py                      # CLI entry point: runs the benchmark and/or scores responses
src/
  llm/runner.py              # Inference loop: prompt sizing, Ollama/LiteLLM calls,
                             # context-window expansion, truncation detection
  pipelines/prompt_generator.py  # Builds system/user prompts per ontology and mode
  pipelines/RAG_a.py         # ChromaDB-backed triple retrieval (OntologyRAG)
  pipelines/TextToSparql.py  # Auxiliary LLM→SPARQL pipeline (used for benchmark construction)
  evaluation/scorer.py       # Metrics: exact match, cosine similarity, BLEU
  reasoner/                  # Owlready2/HermiT ground-truth generation (benchmark construction)
  llm/rdfToTtl.py            # RDF/XML → Turtle conversion utility
data/
  ontologies/                # The five ontologies (see below)
  input/questions.json       # The 71 benchmark questions with SPARQL and ground truth
output/responses/full_run/   # Raw model responses and evaluation matrices (latest run)
analysis/                    # Consolidated numbers extracted from the latest run
chroma_db/                   # Persisted RAG index (regenerated automatically if absent)
```

## Ontologies

| Ontology | File | Size | Notes |
|----------|------|------|-------|
| FOAF | `foaf.rdf` | 44 KB | Fits every context window |
| Wine (W3C OWL Guide) | `wine.rdf` | 78 KB | Fits every context window |
| Pizza | `pizza.owl` | 163 KB | Fits only some models' windows |
| Rientra | `Rientra.ttl` / `Rientra.rdf` | 1.3 / 3.0 MB | Real-world disability-and-occupation KB; two serializations of the same content |
| Gene Ontology | `go.owl` | 130 MB | ~37M tokens; full-paste mode impossible, RAG only |

## Setup

Requires Python 3.12 and a local [Ollama](https://ollama.com) server
(default endpoint `http://localhost:11434`) with the model tags listed above
pulled (all default 4-bit quantizations).

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
```

## Usage

Run the full benchmark (all models, all modes, all ontologies):

```bash
python main.py
```

Useful subsets:

```bash
python main.py --ontology pizza.owl          # one ontology
python main.py --models 1                    # 1 = small models, 2 = large, 3 = all
python main.py --models granite3.3:8b,llama3.1:8b
python main.py --modes Q,Q+Onto+Domain       # subset of prompt modes
python main.py --skip-benchmark              # re-score the latest saved responses only
python main.py --output-dir output/responses/full_run --skip-benchmark
```

Results are written to `output/responses/<timestamp>/<ontology>/`:
`output/` holds one JSON per model × prompt mode with the raw responses
(prompts, answers, token usage, truncation flags), and `comparison/` holds the
evaluation matrix and per-configuration summaries produced by the scorer.

## Scoring conventions

* Metrics are computed on the **short answer** (the text before the `---`
  separator requested by the prompt): exact match (a graded set-overlap score
  for list questions), cosine similarity (`all-MiniLM-L6-v2`), and BLEU.
* A request that yields no parsable short answer is scored **0**, not excluded,
  and counted in `N_Empty`: an absent answer is not an entailed answer.
* Requests whose prompt was silently truncated by the inference server are
  excluded from the averages and flagged (`Context_Truncated`).
* Questions whose ground truth exceeds the 2,048-token generation budget are
  excluded from the main summaries and reported in the `summary_oversized_*`
  files (Q18, Q19, Q25 on Rientra; one GO descendant query).

## A pitfall worth knowing

Ollama (verified on 0.34.2) does **not** reject a prompt longer than the
requested context window `num_ctx`: it silently truncates it to
`num_ctx/2 + 2` tokens, keeping the *tail* of the prompt. The runner therefore
expands `num_ctx` automatically per ontology (bounded by each model's native
window and a safety cap of 98,304 tokens) and flags truncation by comparing
the reported prompt tokens with a lower-bound estimate of the tokens actually
sent. See the comments at the top of `src/llm/runner.py`.

## Reproducing the paper's numbers

The latest complete run is stored in `output/responses/full_run/`
(timestamped 2026-11-23). To recompute all tables from it:

```bash
python main.py --output-dir output/responses/full_run --skip-benchmark
```

The consolidated per-configuration numbers (both the "empty = 0" and the
"answered only" conventions) are in `analysis/numbers_optionC.json`.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
The Rientra ontology is publicly released; the public ontologies (FOAF, Wine,
Pizza, GO) are redistributed or referenced by version and remain under their
respective licenses.
