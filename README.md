# Ensembling Elicitation Strategies for Knowledge Base Construction

This repository represents the implementation for the Automated Knowledge Base Construction shared. The challange task focuses specifically on leveraging Large Language Models to construct structured knowledge bases.

Given a subject s and a relation r, the task is to predict the complete set of correct object strings {o₁, o₂, …, oₖ}, where there may be zero, one or multiple correct answers.

Closed-book knowledge base construction: given a subject entity and a relation,
predict the object entities, using only what a local language model already
knows. Four prompting strategies are run over the same data and combined by a
majority-vote ensemble; `awardWonBy` is handled separately because its answer
lists are one to two orders of magnitude longer than the other relations'.

The proposed system employs multiple prompting strategies and aggregates the resulting predictions according to the relation type in order to generate a final output. Specifically, multiple prediction sets produced by the same model under different prompting strategies are combined to distinguish stable parametric knowledge from prompt-induced hallucinations. The chosen model for this task is Meta's 30B parameter Muse Glimmer.

## Project structure

| Path | Contents |
| --- | --- |
| `data/` | Task splits (`train`, `val`, `test`) in JSON Lines. |
| `prompt_templates/` | Question and cloze prompt templates, one per relation. |
| `scripts/` | The prediction generators and the ensemble. |
| `scripts/common/` | Shared modules: paths, I/O, Ollama access, answer parsing, the CLI and generation loop. |
| `output/` | Generated predictions (not tracked). |
| `evaluate.py` | The official scorer: precision, recall and F1 per relation. |

### Generators

Each script under `scripts/` writes `output/<split>/<script name>.jsonl`.

| Script | Strategy |
| --- | --- |
| `question.py` | Zero-shot, using the question template for the relation. |
| `masked.py` | Zero-shot, using the cloze template with a `[MASK]` for the answer. |
| `fewshot.py` | Few-shot chain-of-thought: worked reasoning examples in a single turn. |
| `least_to_most.py` | Least-to-most: a chain of subquestions, each asked as its own turn. |
| `awards.py` | Recall chaining plus a pruning pass, for `awardWonBy` only. |
| `ensemble.py` | Majority vote over the four strategies, for every relation except `awardWonBy`. |

All generators ask the model to close its reply with an `Answer:` line --
`Answer: <number>` for numeric relations, `Answer: [...]` for list relations --
and share one parser, so the strategies stay directly comparable.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com) running locally; the
generators start `ollama serve` and pull the model themselves if needed.

```bash
pip install -r requirements.txt
```

## Running

Generate predictions for one strategy on one split:

```bash
python scripts/fewshot.py --dataset val
```

Options are the same for every generator:

- `--dataset {train,val,test}` -- which split to run on (default `test`)
- `--model <name>` -- Ollama model to query (default `muse-glimmer:latest`)
- `--limit N` -- only the first N rows per relation, for a smoke test

Run every strategy and ensemble the results:

```bash
./runmodel.sh val
```

This writes `output/val/{question,masked,fewshot,least_to_most,awards}.jsonl`
and `output/predictions/ensemble-val.jsonl`. A complete submission is the
ensemble file plus `output/val/awards.jsonl`, which carries the `awardWonBy`
rows.

## Evaluating

```bash
python evaluate.py -p output/predictions/ensemble-val.jsonl -g data/val.jsonl
```

Any generator's own output can be scored the same way, which is how the
strategies were compared before ensembling.
