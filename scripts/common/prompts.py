"""Prompt-template loading and the answer-format instructions shared by the
zero-shot baselines."""

import csv
from pathlib import Path

ANSWER_FORMAT_NUMERIC = (
    "Always end your reply with a line 'Answer: <number>' containing that "
    "number, no words, no commas, no units."
)

ANSWER_FORMAT_LIST = (
    "Always end your reply with a line 'Answer: [...]' containing a JSON "
    "array of the entity names, or 'Answer: []' if none apply."
)

# Recipient lists run to hundreds of names, too many to survive a JSON array
# within the token budget, so the award generator keeps the same 'Answer:'
# marker but takes one name per line after it.
ANSWER_FORMAT_NAMES = (
    "Always end your reply with a line 'Answer:' followed by the names, one "
    "per line and nothing else, or 'Answer: none' if there are no names."
)


def load_prompt_templates(path: Path) -> dict[str, str]:
    """Read a prompt_templates/*.csv file into {relation: template}."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["Relation"]: row["PromptTemplate"] for row in csv.DictReader(f)}
