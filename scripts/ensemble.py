"""Majority-vote ensemble over the four prompting strategies.

Numeric relations are averaged over the largest cluster of strategies that
agree within a tolerance; list relations keep the items at least two
strategies name. Covers every relation except awardWonBy, whose predictions
come from awards.py alone.
"""

import argparse
import sys

from common.jsonl import load_jsonl, write_jsonl
from common.paths import PREDICTIONS_DIR, REPO_ROOT, SPLITS, output_file
from common.relations import AWARD_RELATION

sys.path.insert(0, str(REPO_ROOT))
from evaluate import normalize_string  # noqa: E402  (the official scorer's own string equality)

STRATEGIES = ("masked", "least_to_most", "question", "fewshot")
TIEBREAK_STRATEGY = "masked"
MIN_VOTES = 2

NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
SINGLE_VALUE_RELATIONS = {"personHasCityOfDeath"}
NUMERIC_TOLERANCE = 0.05


def try_parse_number(value: str) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError, TypeError):
        return None


def format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def numeric_majority(values: dict[str, float]) -> float | None:
    """Mean of the largest cluster of at least two strategies whose numbers are
    mutually within NUMERIC_TOLERANCE, or None if no two agree that closely."""
    best_cluster = None
    for anchor in values:
        cluster = [anchor]
        for other in values:
            if other == anchor:
                continue
            a, b = values[anchor], values[other]
            denom = max(abs(a), abs(b))
            if denom > 0 and abs(a - b) / denom <= NUMERIC_TOLERANCE:
                cluster.append(other)
        if len(cluster) >= MIN_VOTES and (best_cluster is None or len(cluster) > len(best_cluster)):
            best_cluster = cluster

    if best_cluster is None:
        return None
    return sum(values[name] for name in best_cluster) / len(best_cluster)


def ensemble_numeric(per_strategy: dict[str, list[str]]) -> list[str]:
    votes = {}
    for strategy, objects in per_strategy.items():
        if objects:
            number = try_parse_number(objects[0])
            if number is not None:
                votes[strategy] = number

    majority = numeric_majority(votes)
    if majority is not None:
        return [format_number(majority)]
    return list(per_strategy[TIEBREAK_STRATEGY])


def clean_surface(value: str) -> str:
    """Cosmetic cleanup of the surface form kept in the output; normalize_string
    is used only for the equality check, since it lowercases and strips diacritics."""
    return value.strip().strip("\"'").strip(" .,!?").strip()


def majority_vote_items(per_strategy: dict[str, list[str]]) -> list[str]:
    """Items named by at least MIN_VOTES strategies, most-agreed first. Reaching
    no majority yields [], which is the expected answer for many subjects."""
    votes = {}
    surface = {}
    order = []
    for objects in per_strategy.values():
        seen = set()
        for raw in objects:
            key = normalize_string(raw)
            if not key or key in seen:
                continue  # a strategy naming the same item twice is still one vote
            seen.add(key)
            votes[key] = votes.get(key, 0) + 1
            if key not in surface:
                surface[key] = clean_surface(raw)
                order.append(key)

    winners = [key for key in order if votes[key] >= MIN_VOTES]
    winners.sort(key=lambda key: (-votes[key], order.index(key)))
    return [surface[key] for key in winners]


def ensemble_row(relation: str, per_strategy: dict[str, list[str]]) -> list[str]:
    if relation in NUMERIC_RELATIONS:
        return ensemble_numeric(per_strategy)

    winners = majority_vote_items(per_strategy)
    if relation in SINGLE_VALUE_RELATIONS:
        # A person has exactly one city of death; keep only the strongest
        # consensus if more than one somehow reaches quorum.
        return winners[:1]
    return winners


def load_strategy_rows(split: str) -> dict[str, dict[tuple[str, str], list[str]]]:
    """Each strategy's predictions indexed by (SubjectEntity, Relation), so the
    files need not list rows in the same order."""
    indexed = {}
    for strategy in STRATEGIES:
        path = output_file(split, strategy)
        if not path.exists():
            raise SystemExit(f"Missing predictions: {path}. Run `python scripts/{strategy}.py "
                             f"--dataset {split}` first.")

        by_key = {}
        for row in load_jsonl(path):
            if row["Relation"] == AWARD_RELATION:
                continue
            key = (row["SubjectEntity"], row["Relation"])
            if key in by_key and by_key[key] != row["ObjectEntities"]:
                raise SystemExit(f"{path} has conflicting duplicate rows for {key}: "
                                 f"{by_key[key]} vs {row['ObjectEntities']}.")
            by_key[key] = row["ObjectEntities"]
        indexed[strategy] = by_key
    return indexed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="test", choices=SPLITS,
                        help="Which split's predictions to ensemble")
    args = parser.parse_args()

    per_strategy = load_strategy_rows(args.dataset)

    key_sets = {strategy: set(rows) for strategy, rows in per_strategy.items()}
    all_keys = set.union(*key_sets.values())
    incomplete = {strategy: len(all_keys - keys) for strategy, keys in key_sets.items() if all_keys - keys}
    if incomplete:
        raise SystemExit("Strategy files do not cover the same (SubjectEntity, Relation) rows. "
                         f"Rows missing per strategy: {incomplete}")

    ordered_keys = list(per_strategy[STRATEGIES[0]])
    ensembled = [
        {
            "SubjectEntity": subject,
            "Relation": relation,
            "ObjectEntities": ensemble_row(relation, {s: per_strategy[s][(subject, relation)] for s in STRATEGIES}),
        }
        for subject, relation in ordered_keys
    ]

    path = PREDICTIONS_DIR / f"ensemble-{args.dataset}.jsonl"
    write_jsonl(path, ensembled)
    print(f"Ensembled {len(ensembled)} rows from {list(STRATEGIES)} into {path}")


if __name__ == "__main__":
    main()
