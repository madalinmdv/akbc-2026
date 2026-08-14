"""Recall-chained awardWonBy predictions.

The only generator for awardWonBy, whose object lists run to hundreds of
recipients rather than the handful the other relations expect. It asks for
recipients repeatedly, each round shown the names already gathered so it only
has to add what is missing, then prunes the merged list of the names the model
is certain are wrong.
"""

from pathlib import Path

from tqdm import tqdm

from common.ollama_client import build_system_prompt, call_model
from common.parsing import parse_name_list
from common.prompts import ANSWER_FORMAT_NAMES
from common.relations import AWARD_RELATION
from common.runner import build_parser, load_rows, run_generation

AWARD_MAX_TOKENS = 4096
AWARD_TEMPERATURE = 0.15

MAX_ROUNDS = 6                  # maximum recall expansion rounds per award
MIN_NEW_NAMES = 2               # stop chaining once a round adds fewer than this
ROUNDS_BEFORE_DIMINISHING = 2   # ...but only from this round onward

# Chaining is tuned for recall, so later rounds drag in names the model was
# reaching for. The pruning pass hands the merged list back in chunks and asks
# only for the names it is certain never received the award; the default is to
# keep, so hedged or unparseable replies cost nothing.
PRUNE_CHUNK_SIZE = 15
PRUNE_MAX_TOKENS = 512
PRUNE_TEMPERATURE = 0.0

SYSTEM_PROMPT_AWARD = (
    "You are a comprehensive awards knowledge base. You know the recipients "
    "of awards and prizes across all decades. Awards often have dozens or "
    "hundreds of recipients, so do not stop after the most famous names: "
    "include early recipients, recent recipients, and lesser-known ones. "
    "Provide only the requested information without explanations, uncertainty "
    "statements, or additional context. "
) + ANSWER_FORMAT_NAMES

FIRST_ROUND_QUERY = (
    "List ALL known recipients of {award}. Include winners from every decade: "
    "early winners, recent winners, and any lesser-known recipients."
)

FOLLOW_UP_QUERY = (
    "Recipients of {award} identified so far:\n{listed}\n\n"
    "Who ELSE has received {award} that is NOT in the list above? "
    "Only add names you are genuinely confident about, and do not repeat any "
    "name already listed. If you cannot think of any more, reply with exactly: "
    "Answer: none"
)

SYSTEM_PROMPT_PRUNE = (
    "You are a strict fact-checker for award recipients. You are shown a "
    "candidate list and you name only the ones you are EXTREMELY confident "
    "never received the award -- not the ones you merely cannot confirm or "
    "are unsure about, which you leave alone. Provide only the requested "
    "information without explanations, uncertainty statements, or additional "
    "context. "
) + ANSWER_FORMAT_NAMES

PRUNE_QUERY = (
    "Award: {award}\n\n"
    "Candidates:\n{listed}\n\n"
    "Which of the candidates above are you EXTREMELY confident never received "
    "{award}? List only those names, copied exactly as written above. "
    "Leave out any name you are merely unsure about or cannot confirm -- only "
    "name the ones you are certain are wrong. If every candidate above is "
    "plausible, reply with exactly: Answer: none"
)


def build_round_query(award: str, round_num: int, merged: list[str]) -> str:
    """The opening round asks cold; later rounds show the model what it has
    already committed to and ask only for what is missing."""
    if round_num == 0:
        return FIRST_ROUND_QUERY.format(award=award)
    return FOLLOW_UP_QUERY.format(award=award, listed="\n".join(merged))


def prune_candidates(model_id: str, award: str, candidates: list[str]) -> list[str]:
    """Drop the names the model is certain never received the award, keeping
    the original order.

    A rejection counts only when it matches a candidate in the chunk verbatim
    (case-insensitively), so a hedge, an invented name, or a mangled echo
    leaves the list untouched."""
    if not candidates:
        return []

    system_prompt = build_system_prompt(model_id, SYSTEM_PROMPT_PRUNE)
    rejected = set()

    for start in range(0, len(candidates), PRUNE_CHUNK_SIZE):
        chunk = candidates[start:start + PRUNE_CHUNK_SIZE]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PRUNE_QUERY.format(award=award, listed="\n".join(chunk))},
        ]
        try:
            raw_answer = call_model(model_id, messages, temperature=PRUNE_TEMPERATURE,
                                    num_predict=PRUNE_MAX_TOKENS)
        except Exception as e:
            chunk_num = start // PRUNE_CHUNK_SIZE + 1
            print(f"[query error] {model_id} / {award} (prune chunk {chunk_num}): {e}")
            continue

        in_chunk = {name.casefold() for name in chunk}
        chunk_rejected = {n.casefold() for n in parse_name_list(raw_answer) if n.casefold() in in_chunk}

        # A chunk rejected wholesale is far more likely a model echoing the
        # list back than genuine certainty about every name in it.
        if len(chunk_rejected) == len(chunk):
            continue

        rejected |= chunk_rejected

    kept = [name for name in candidates if name.casefold() not in rejected]
    if rejected:
        tqdm.write(f"{award}: pruned {len(candidates) - len(kept)}/{len(candidates)} candidates")
    return kept


def predict(model_id: str, subject: str, relation: str) -> list[str]:
    """Chain recall rounds for one award, merging and deduping the names
    recovered from every round, then prune the certain non-recipients."""
    system_prompt = build_system_prompt(model_id, SYSTEM_PROMPT_AWARD)
    seen = set()
    merged = []

    for round_num in range(MAX_ROUNDS):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_round_query(subject, round_num, merged)},
        ]
        try:
            raw_answer = call_model(model_id, messages, temperature=AWARD_TEMPERATURE,
                                    num_predict=AWARD_MAX_TOKENS)
        except Exception as e:
            print(f"[query error] {model_id} / {subject} (round {round_num + 1}): {e}")
            break

        # A round that only repeats known names is as much a stopping signal as
        # an outright "none".
        fresh = []
        for name in parse_name_list(raw_answer):
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                fresh.append(name)

        if not fresh:
            break

        merged.extend(fresh)

        if round_num >= ROUNDS_BEFORE_DIMINISHING and len(fresh) < MIN_NEW_NAMES:
            break

    return prune_candidates(model_id, subject, merged)


def main():
    args = build_parser(__doc__).parse_args()
    rows = load_rows(args.dataset, {AWARD_RELATION}, args.limit)
    run_generation(Path(__file__).stem, args.model, rows, args.dataset, predict)


if __name__ == "__main__":
    main()
