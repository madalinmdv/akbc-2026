"""The command-line interface and generation loop shared by every generator."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from .jsonl import load_jsonl, write_jsonl
from .ollama_client import DEFAULT_MODEL, ensure_model_pulled, ensure_server_running
from .paths import SPLITS, dataset_file, output_file

Predictor = Callable[[str, str, str], list[str]]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="test", choices=SPLITS,
                        help="Which split under data/ to run on")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model to query")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows per relation (smoke test)")
    return parser


def load_rows(split: str, relations: Iterable[str], limit: int | None = None) -> list[dict]:
    """Rows of `split` for the given relations. `limit` caps rows per relation
    rather than overall, so a small smoke test still covers each of them."""
    allowed = set(relations)
    rows = [row for row in load_jsonl(dataset_file(split)) if row["Relation"] in allowed]
    if not limit:
        return rows

    kept = Counter()
    limited = []
    for row in rows:
        if kept[row["Relation"]] < limit:
            limited.append(row)
            kept[row["Relation"]] += 1
    return limited


def run_generation(name: str, model_id: str, rows: list[dict], split: str,
                   predict: Predictor) -> Path:
    """Query `predict` for every row and write output/<split>/<name>.jsonl."""
    ensure_server_running()
    ensure_model_pulled(model_id)

    print(f"{name}: {len(rows)} rows from {split}.jsonl with {model_id}")

    predictions = []
    empty = 0
    progress = tqdm(rows, desc=name)
    for row in progress:
        objects = predict(model_id, row["SubjectEntity"], row["Relation"])
        if not objects:
            empty += 1
        progress.set_postfix({"empty": empty})
        predictions.append({
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": objects,
        })

    path = output_file(split, name)
    write_jsonl(path, predictions)
    print(f"{name}: wrote {len(predictions)} predictions to {path} ({empty} empty)")
    return path
