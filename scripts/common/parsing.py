"""Turning a model reply into the list of object entities.

Every generator instructs the model to close its reply with an `Answer:` line
-- `Answer: <number>` for numeric relations, `Answer: [...]` for list
relations -- so `parse_answer` serves all of them. `parse_name_list` handles
the one variant that cannot use a JSON array: the award recall chain, whose
replies run to hundreds of names and are read line by line.
"""

import json
import re

NONE_LIKE = {
    "none", "n/a", "na", "unknown", "not applicable", "nothing",
    "no names", "no recipients", "no winners",
}

# Hedging and refusals that would otherwise be read as entity names.
HEDGE_PHRASE_RE = re.compile(
    r"\b(i don'?t know|i do not know|i'?m not sure|i am not sure|not aware|"
    r"cannot (?:confirm|recall|provide|determine|find)|"
    r"could(?:n'?t| not) (?:find|recall|confirm|determine)|"
    r"no information|unable to|as an ai|i cannot|sorry|don'?t have)\b",
    re.IGNORECASE,
)

MAX_NAME_WORDS = 8
MAX_NAME_CHARS = 100


def strip_think_tags(raw_response: str) -> str:
    """Drop <think> reasoning blocks, keeping newlines intact for line-based parsing."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.replace("</think>", "")


def clean_model_response(raw_response: str) -> str:
    """Drop <think> blocks and collapse all whitespace to single spaces."""
    return " ".join(strip_think_tags(raw_response).split()).strip()


def extract_answer_segment(text: str) -> str:
    """The text after the last `Answer:` marker, or all of it if there is none."""
    matches = list(re.finditer(r"Answer\s*:", text, re.IGNORECASE))
    if not matches:
        return text
    return text[matches[-1].end():].strip()


def extract_number(text: str) -> str:
    if not text:
        return ""
    # Comma-grouped numbers first, so "60,206" is not truncated to "60".
    grouped = re.search(r"\b\d{1,3}(?:,\d{3})+\b", text)
    if grouped:
        return grouped.group().replace(",", "")
    plain = re.search(r"\b\d+\.?\d*\b", text)
    return plain.group() if plain else ""


def parse_list_answer(segment: str) -> list[str]:
    """A JSON array from the answer segment, falling back to a comma split."""
    bracketed = re.search(r"\[.*\]", segment, re.DOTALL)
    raw = bracketed.group(0) if bracketed else segment
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    inner = raw.strip().strip("[]")
    parts = [part.strip(" \"'") for part in inner.split(",")]
    return [part for part in parts if part and part.lower() not in NONE_LIKE]


def parse_answer(raw_response: str, answer_type: str) -> list[str]:
    segment = extract_answer_segment(clean_model_response(raw_response))
    if answer_type == "numeric":
        number = extract_number(segment)
        return [number] if number else []
    return parse_list_answer(segment)


def parse_name_list(raw_response: str) -> list[str]:
    """Names from a reply that lists them one per line, comma-separated, or as
    a JSON array, with or without a closing `Answer:` marker. Unlike
    `parse_answer` this preserves newlines, so long lists that a JSON array
    would not survive still split correctly."""
    segment = extract_answer_segment(strip_think_tags(raw_response)).strip().strip("[]")
    if not segment:
        return []

    names = []
    for part in re.split(r"[,;\n]", segment):
        name = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip(" \t\"'*_.-")
        name = " ".join(name.split())
        if not name or name.lower() in NONE_LIKE:
            continue
        if HEDGE_PHRASE_RE.search(name) or len(name.split()) > MAX_NAME_WORDS:
            continue
        if 1 < len(name) < MAX_NAME_CHARS:
            names.append(name)
    return names
