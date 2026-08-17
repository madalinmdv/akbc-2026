# Ensembling Elicitation Strategies for Knowledge Base Construction

Large language models represent a promising alternative for automating structured knowledge bases construction, but their reliance on parametric knowledge makes them susceptible to hallucinations and incomplete information. The Automated Knowledge Base Construction competition investigates the use of large language models to construct structured knowledge bases from their parametric knowledge alone, without fine-tuning or external retrieval. Given a subject--relation pair, participants must predict the complete set of correct objects across several relation types.

| Relation | Description | Cardinality |
|---|---|---|
| `awardWonBy` | Identifies the recipients of a specified award | Often high — many awards are conferred upon dozens or hundreds of entities over time |
| `countryLandBorderCountries` | Links a country to the nations sharing its land border | Variable — empty for borderless countries, multiple values otherwise |
| `hasArea` | Specifies a geographic entity's surface area in square kilometres | Single value |
| `companyTradesAtStockExchange` | Maps a company to the stock exchange(s) where its shares are publicly traded | Variable — empty for unlisted subsidiaries, multiple values otherwise |
| `hasCapacity` | Denotes the maximum spectator capacity of a venue | Single value |
| `personHasCityOfDeath` | Records the city where an individual passed away | Empty if the person is still living |

Our approach proposes an ensemble of four prompting strategies: **question-style, masked-style, few-shot chain-of-thought, and least-to-most prompting.** It aggregates their predictions using relation-aware majority voting. There is an exception however for the *awardWonBy* relation, where the large number of correct objects per subject makes voting across strategies too restrictive. For this relation a **self-refinement** The resulting system achieves a Macro-F1 of **0.6651** on the hidden test set, substantially outperforming the provided baseline of **0.294**.

<img width="2613" height="2613" alt="3be2d7e9-7948-473c-bb77-d31405d11817_page-0001" src="https://github.com/user-attachments/assets/993d05b0-d868-4809-ae32-e7b6f3d704b5" />


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
