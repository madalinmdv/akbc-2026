"""Repository locations, resolved from this file so scripts run from any cwd."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
PROMPT_TEMPLATE_DIR = REPO_ROOT / "prompt_templates"
OUTPUT_DIR = REPO_ROOT / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

SPLITS = ("train", "val", "test")


def dataset_file(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def output_file(split: str, name: str) -> Path:
    """Prediction file for one generator on one split, e.g. output/val/fewshot.jsonl."""
    return OUTPUT_DIR / split / f"{name}.jsonl"


def prompt_template_file(name: str) -> Path:
    return PROMPT_TEMPLATE_DIR / name
