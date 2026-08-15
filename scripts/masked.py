"""Zero-shot masked prompting."""

from pathlib import Path

from common.ollama_client import build_system_prompt, call_model
from common.parsing import parse_answer
from common.paths import prompt_template_file
from common.prompts import ANSWER_FORMAT_LIST, ANSWER_FORMAT_NUMERIC, load_prompt_templates
from common.relations import AWARD_RELATION, answer_type_for
from common.runner import build_parser, load_rows, run_generation

PROMPTS_FILE = prompt_template_file("masked_prompts.csv")
MASK_TOKEN = "[MASK]"

SYSTEM_PROMPT_NUMERIC = (
    "You are a precise closed-book estimation assistant. You will be given a "
    "sentence with a blank marked '[MASK]'. Determine the single best-estimate "
    "number that fills the blank -- never refuse or say you don't know, even "
    "if you are not certain of the exact figure. "
) + ANSWER_FORMAT_NUMERIC

SYSTEM_PROMPT_LIST = (
    "You are a precise closed-book factual assistant. You will be given a "
    "sentence with a blank marked '[MASK]' that may stand for zero, one, or "
    "several entities. Determine every entity that correctly fills the blank, "
    "drawing only on what you actually know -- if none apply, the answer is "
    "empty; do not guess. "
) + ANSWER_FORMAT_LIST


def predict(model_id: str, template: str, subject: str, relation: str) -> list[str]:
    answer_type = answer_type_for(relation)
    system_prompt = SYSTEM_PROMPT_NUMERIC if answer_type == "numeric" else SYSTEM_PROMPT_LIST
    sentence = template.format(subject_entity=subject, mask_token=MASK_TOKEN)
    messages = [
        {"role": "system", "content": build_system_prompt(model_id, system_prompt)},
        {"role": "user", "content": sentence},
    ]

    try:
        raw_answer = call_model(model_id, messages)
    except Exception as e:
        print(f"[query error] {model_id} / {subject} ({relation}): {e}")
        return []

    return parse_answer(raw_answer, answer_type)


def main():
    args = build_parser(__doc__).parse_args()
    templates = load_prompt_templates(PROMPTS_FILE)
    relations = set(templates) - {AWARD_RELATION}
    rows = load_rows(args.dataset, relations, args.limit)

    run_generation(
        Path(__file__).stem, args.model, rows, args.dataset,
        lambda model_id, subject, relation: predict(model_id, templates[relation], subject, relation),
    )


if __name__ == "__main__":
    main()
