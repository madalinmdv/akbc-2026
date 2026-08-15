from pathlib import Path

from common.ollama_client import build_system_prompt, call_model
from common.parsing import clean_model_response, parse_answer
from common.prompts import ANSWER_FORMAT_LIST, ANSWER_FORMAT_NUMERIC
from common.relations import QUESTIONS, answer_type_for
from common.runner import build_parser, load_rows, run_generation

LTM_TEMPERATURE = 0.05

# Subanswers are scaffolding, not essays; a tight budget also keeps the
# accumulated context small.
SUBANSWER_MAX_TOKENS = 400

# Every subanswer is fed back into the conversation, so an empty one is
# contagious. Retry once with more temperature, then drop the turn entirely
# rather than leave a placeholder for later turns to imitate.
EMPTY_RETRY_TEMPERATURE = 0.3


def render_numeric_ltm(question_template: str, subquestions: list[str], examples: list[dict]) -> str:
    """Render {"entity", "sub_answers", "answer"} examples into the same
    Problem/Subquestion/Subanswer/Answer blocks the list relations use verbatim."""
    blocks = []
    for example in examples:
        problem = question_template.format(subject=example["entity"])
        lines = [f"Problem: {problem}"]
        for i, (subquestion, subanswer) in enumerate(zip(subquestions, example["sub_answers"]), 1):
            lines.append(f"Subquestion {i}: {subquestion.format(subject=example['entity'])}")
            lines.append(f"Subanswer {i}: {subanswer}")
        lines.append(f"Final question: {problem}")
        lines.append(f"Answer: {example['answer']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# --- hasArea ---------------------------------------------------------------

SYSTEM_PROMPT_AREA = (
    "You are a precise closed-book estimation assistant. You solve area "
    "questions least-to-most: you are asked a series of simpler subquestions "
    "first, one at a time, and you answer each one in one or two sentences, "
    "using your earlier answers. Only at the end are you asked the original "
    "question, and there you must commit to a single best-estimate numeric "
    "area in square kilometers -- never refuse or say you don't know, even if "
    "you are not certain of the exact figure."
)

SUBQUESTIONS_AREA = [
    "What kind of entity is {subject} (lake, island, country, etc.) and where is it?",
    "Compared with other entities of that same kind, and with entities of well-known scale, roughly how large is {subject}?",
]

LTM_AREA = [
    {
        "entity": "Moraine Lake in Banff National Park",
        "sub_answers": [
            "Moraine Lake is a glacially-fed lake in Banff National Park in the "
            "Canadian Rockies, known for its turquoise water and surrounded by the "
            "Valley of the Ten Peaks.",
            "It is a small alpine lake, not a major freshwater body like the nearby "
            "Lake Louise or a large reservoir -- alpine lakes of this kind typically "
            "cover well under a square kilometer.",
        ],
        "answer": "0.5",
    },
    {
        "entity": "Nantucket",
        "sub_answers": [
            "Nantucket is an island and county off the coast of Massachusetts, in "
            "the northeastern United States.",
            "It is a mid-sized offshore island, comparable in scale to other "
            "well-known New England islands such as Martha's Vineyard, rather than a "
            "small islet or a large landmass -- such islands are typically on the "
            "order of a hundred or so square kilometers.",
        ],
        "answer": "123.8",
    },
    {
        "entity": "Nigeria",
        "sub_answers": [
            "Nigeria is a country in West Africa, the most populous country on the "
            "continent, bordering Niger, Chad, Cameroon, and Benin.",
            "It is a large national territory spanning multiple climate zones from "
            "coastal mangroves to Sahelian savanna -- countries of this scale "
            "typically span several hundred thousand square kilometers.",
        ],
        "answer": "923768",
    },
]


# --- hasCapacity -----------------------------------------------------------

SYSTEM_PROMPT_CAPACITY = (
    "You are a precise closed-book estimation assistant. You solve venue "
    "capacity questions least-to-most: you are asked a series of simpler "
    "subquestions first, one at a time, and you answer each one in one or two "
    "sentences, using your earlier answers. Only at the end are you asked the "
    "original question, and there you must commit to a single best-estimate "
    "numeric capacity -- never refuse or say you don't know, even if you are "
    "not certain of the exact figure."
)

SUBQUESTIONS_CAPACITY = [
    "What kind of venue is {subject}, and what geographic or institutional context does it serve (small local/municipal ground, collegiate venue, major professional or national venue)?",
    "Which comparable venues of that same type and tier do you know, and roughly how many spectators do they hold?",
]

LTM_CAPACITY = [
    {
        "entity": "Sungui Sports Complex in Incheon",
        "sub_answers": [
            "Sungui Sports Complex is a multi-purpose athletic venue in Incheon, "
            "South Korea, serving a large city rather than a small municipality.",
            "Incheon has hosted major multi-sport events including the 2014 Asian "
            "Games and maintains several mid-to-large stadiums; a city-level "
            "multi-purpose sports complex used for athletics and football typically "
            "holds tens of thousands of spectators.",
        ],
        "answer": "35000",
    },
    {
        "entity": "John A. Farrell Stadium in Pennsylvania",
        "sub_answers": [
            "John A. Farrell Stadium in Pennsylvania is associated with collegiate "
            "and scholastic athletics and football.",
            "Venues serving a college or school district, rather than a major "
            "professional franchise, usually hold a few thousand spectators, sized to "
            "the local community rather than a mass audience.",
        ],
        "answer": "7500",
    },
    {
        "entity": "Estadio Revolución Ciudad de Guatemala in Guatemala City",
        "sub_answers": [
            "Estadio Revolución is a stadium in Guatemala City, Guatemala, whose name "
            "points to a specific institutional venue rather than the country's "
            "largest national stadium.",
            "Secondary or institutional stadiums in a capital, used for smaller "
            "matches, typically have modest capacities compared with a national "
            "flagship stadium such as Estadio Mateo Flores.",
        ],
        "answer": "5000",
    },
]


# --- countryLandBordersCountry ---------------------------------------------

SYSTEM_PROMPT_BORDERS = (
    "You are a precise geography assistant answering closed-book. You solve "
    "border questions least-to-most: you are asked a series of simpler "
    "subquestions first, one at a time, and you answer each one in one or two "
    "sentences, using your earlier answers. Only at the end are you asked the "
    "original question, and there you must give the full set of neighbors -- "
    "or an empty list for a country with no land borders at all."
)

SUBQUESTIONS_BORDERS = [
    "What is {subject}, and on which continent or in which region is it located?",
    "Can {subject} have land borders at all, or is it an island nation or territory surrounded entirely by water?",
    "Going around its frontier, which countries lie immediately adjacent to {subject} on each side?",
]

FEW_SHOT_BORDERS = """Problem: Which countries share a land border with New Zealand?
Subquestion 1: What is New Zealand, and on which continent or in which region is it located?
Subanswer 1: New Zealand is an island country in Oceania, in the southwestern Pacific Ocean, made up of the North Island, South Island, and numerous smaller islands.
Subquestion 2: Can New Zealand have land borders at all, or is it an island nation or territory surrounded entirely by water?
Subanswer 2: As an island nation entirely surrounded by ocean, it has no shared land boundary with any other country.
Subquestion 3: Going around its frontier, which countries lie immediately adjacent to New Zealand on each side?
Subanswer 3: There is no adjoining country on any side, so there are no neighbors to name.
Final question: Which countries share a land border with New Zealand?
Answer: []

Problem: Which countries share a land border with Germany?
Subquestion 1: What is Germany, and on which continent or in which region is it located?
Subanswer 1: Germany is a country in Central Europe.
Subquestion 2: Can Germany have land borders at all, or is it an island nation or territory surrounded entirely by water?
Subanswer 2: It is a continental country with an extensive land frontier, not an island or enclave, so it does have land borders.
Subquestion 3: Going around its frontier, which countries lie immediately adjacent to Germany on each side?
Subanswer 3: Denmark to the north; Poland and the Czech Republic to the east; Austria and Switzerland to the south; France, Luxembourg, Belgium, and the Netherlands to the west.
Final question: Which countries share a land border with Germany?
Answer: ["Austria", "Belgium", "Czech Republic", "Denmark", "France", "Luxembourg", "Netherlands", "Poland", "Switzerland"]

Problem: Which countries share a land border with Sierra Leone?
Subquestion 1: What is Sierra Leone, and on which continent or in which region is it located?
Subanswer 1: Sierra Leone is a country on the Atlantic coast of West Africa.
Subquestion 2: Can Sierra Leone have land borders at all, or is it an island nation or territory surrounded entirely by water?
Subanswer 2: It is a continental West African country, not an island, so it does have land borders.
Subquestion 3: Going around its frontier, which countries lie immediately adjacent to Sierra Leone on each side?
Subanswer 3: Guinea to the north and east, and Liberia to the south and east; the remaining frontier is Atlantic coastline.
Final question: Which countries share a land border with Sierra Leone?
Answer: ["Guinea", "Liberia"]
"""


# --- personHasCityOfDeath --------------------------------------------------

SYSTEM_PROMPT_DEATH = (
    "You identify the city where a person died, answering closed-book. You "
    "solve this least-to-most: you are asked a series of simpler subquestions "
    "first, one at a time, and you answer each one in one or two sentences, "
    "using your earlier answers. Be careful NOT to confuse the city a person "
    "is most FAMOUS for, or lived or worked in, with the city where they "
    "actually died -- these are frequently different. Only at the end are you "
    "asked the original question, and there you give a single city at city "
    "granularity, or an empty list if the person is alive or the death city "
    "cannot be established."
)

SUBQUESTIONS_DEATH = [
    "Who is or was {subject}, and has {subject} died?",
    "Which city is {subject} most famous for, or most associated with through living or working there?",
    "If {subject} has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?",
]

FEW_SHOT_DEATH = """Problem: In which city did Sigmund Freud die?
Subquestion 1: Who is or was Sigmund Freud, and has Sigmund Freud died?
Subanswer 1: Sigmund Freud was the founder of psychoanalysis; he died in 1939 and is deceased.
Subquestion 2: Which city is Sigmund Freud most famous for, or most associated with through living or working there?
Subanswer 2: He is overwhelmingly associated with Vienna, where he lived and worked for most of his life.
Subquestion 3: If Sigmund Freud has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?
Subanswer 3: He fled Vienna in 1938 after the Nazi annexation of Austria and died the following year in London, so the death city differs from the famous city.
Final question: In which city did Sigmund Freud die?
Answer: ["London"]

Problem: In which city did Uwe Timm die?
Subquestion 1: Who is or was Uwe Timm, and has Uwe Timm died?
Subanswer 1: Uwe Timm is a German author who is still living as of current records.
Subquestion 2: Which city is Uwe Timm most famous for, or most associated with through living or working there?
Subanswer 2: He is associated with Munich, where he has long lived and worked.
Subquestion 3: If Uwe Timm has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?
Subanswer 3: Since he has not died, there is no city of death to identify, and the city he lives in must not be reported as one.
Final question: In which city did Uwe Timm die?
Answer: []

Problem: In which city did Albert Einstein die?
Subquestion 1: Who is or was Albert Einstein, and has Albert Einstein died?
Subanswer 1: Albert Einstein was a theoretical physicist; he died in 1955 and is deceased.
Subquestion 2: Which city is Albert Einstein most famous for, or most associated with through living or working there?
Subanswer 2: He is famous for his work in Germany and Switzerland, particularly Berlin and Zurich.
Subquestion 3: If Albert Einstein has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?
Subanswer 3: He spent his final years in the United States and died in Princeton, New Jersey, a different city from the ones he is famous for.
Final question: In which city did Albert Einstein die?
Answer: ["Princeton"]

Problem: In which city did Henk van Kerkwijk die?
Subquestion 1: Who is or was Henk van Kerkwijk, and has Henk van Kerkwijk died?
Subanswer 1: I cannot confidently verify whether Henk van Kerkwijk has died, nor locate reliable records of a death.
Subquestion 2: Which city is Henk van Kerkwijk most famous for, or most associated with through living or working there?
Subanswer 2: No city can be reliably associated with him either.
Subquestion 3: If Henk van Kerkwijk has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?
Subanswer 3: Since the death cannot be established, no specific city of death can be identified, and a guess would not be appropriate.
Final question: In which city did Henk van Kerkwijk die?
Answer: []

Problem: In which city did Milan Lasica die?
Subquestion 1: Who is or was Milan Lasica, and has Milan Lasica died?
Subanswer 1: Milan Lasica was a Slovak actor and writer, and he has died.
Subquestion 2: Which city is Milan Lasica most famous for, or most associated with through living or working there?
Subanswer 2: He lived and worked in Bratislava, where his theatre career was based.
Subquestion 3: If Milan Lasica has died, in which city did the death actually take place, and is that the same city as the one just named or a different one?
Subanswer 3: He also died in Bratislava; here the death city and the lived city happen to coincide.
Final question: In which city did Milan Lasica die?
Answer: ["Bratislava"]
"""


# --- companyTradesAtStockExchange ------------------------------------------

SYSTEM_PROMPT_STOCK = (
    "You are a precise financial-facts assistant answering closed-book. You "
    "solve listing questions least-to-most: you are asked a series of simpler "
    "subquestions first, one at a time, and you answer each one in one or two "
    "sentences, using your earlier answers. Only at the end are you asked the "
    "original question, and there you name every exchange where the company's "
    "own shares trade -- or an empty list if it is not publicly traded."
)

SUBQUESTIONS_STOCK = [
    "What is {subject} -- its country, sector, and ownership structure?",
    "Is {subject} publicly listed in its own right, or is it privately held, a wholly-owned subsidiary not separately listed, or delisted?",
    "If it is listed, on which exchange do its own shares primarily trade, and does it also have secondary, dual, or ADR/GDR listings elsewhere?",
]

FEW_SHOT_STOCK = """Problem: On which stock exchanges does Bangladesh Cement Manufacturers Association trade?
Subquestion 1: What is Bangladesh Cement Manufacturers Association -- its country, sector, and ownership structure?
Subanswer 1: The Bangladesh Cement Manufacturers Association (BCMA) is a trade/industry association representing cement producers in Bangladesh, not a commercial corporation or operating company.
Subquestion 2: Is Bangladesh Cement Manufacturers Association publicly listed in its own right, or is it privately held, a wholly-owned subsidiary not separately listed, or delisted?
Subanswer 2: As a membership-based industry body, it does not issue equity shares to the public and is therefore not publicly listed on any exchange.
Subquestion 3: If it is listed, on which exchange do its own shares primarily trade, and does it also have secondary, dual, or ADR/GDR listings elsewhere?
Subanswer 3: Since it is not a publicly traded entity, there are no primary or secondary listings to name.
Final question: On which stock exchanges does Bangladesh Cement Manufacturers Association trade?
Answer: []

Problem: On which stock exchanges does West Japan Railway Company trade?
Subquestion 1: What is West Japan Railway Company -- its country, sector, and ownership structure?
Subanswer 1: West Japan Railway Company (JR West) is a Japanese transportation company operating passenger and freight railways across the Kansai, Chugoku, and Shikoku regions; it was formed during the privatization of the former state-owned Japanese National Railways and operates as an independent publicly held corporation.
Subquestion 2: Is West Japan Railway Company publicly listed in its own right, or is it privately held, a wholly-owned subsidiary not separately listed, or delisted?
Subanswer 2: The company is publicly listed in its own right; it conducted its initial public offering in October 1991 and remains actively traded on the open market.
Subquestion 3: If it is listed, on which exchange do its own shares primarily trade, and does it also have secondary, dual, or ADR/GDR listings elsewhere?
Subanswer 3: Its shares trade on the Tokyo Stock Exchange (Prime Market, ticker 9021), and it does not maintain active secondary domestic cross-listings or any ADR/GDR programs on foreign exchanges.
Final question: On which stock exchanges does West Japan Railway Company trade?
Answer: ["Tokyo Stock Exchange"]

Problem: On which stock exchanges does Energen trade?
Subquestion 1: What is Energen -- its country, sector, and ownership structure?
Subanswer 1: Energen Corporation is a United States-based energy and utility company headquartered in Birmingham, Alabama, structured as a publicly traded corporation.
Subquestion 2: Is Energen publicly listed in its own right, or is it privately held, a wholly-owned subsidiary not separately listed, or delisted?
Subanswer 2: The company is publicly listed and actively trades on the open market; it is not privately held or a non-listed subsidiary.
Subquestion 3: If it is listed, on which exchange do its own shares primarily trade, and does it also have secondary, dual, or ADR/GDR listings elsewhere?
Subanswer 3: Energen's common stock trades on the New York Stock Exchange under the ticker symbol EGN, with no secondary, dual, or ADR/GDR listings on other exchanges.
Final question: On which stock exchanges does Energen trade?
Answer: ["New York Stock Exchange"]
"""


RELATIONS = {
    "hasArea": {
        "system_prompt": SYSTEM_PROMPT_AREA,
        "subquestions": SUBQUESTIONS_AREA,
        "few_shot": render_numeric_ltm(QUESTIONS["hasArea"], SUBQUESTIONS_AREA, LTM_AREA),
    },
    "hasCapacity": {
        "system_prompt": SYSTEM_PROMPT_CAPACITY,
        "subquestions": SUBQUESTIONS_CAPACITY,
        "few_shot": render_numeric_ltm(QUESTIONS["hasCapacity"], SUBQUESTIONS_CAPACITY, LTM_CAPACITY),
    },
    "countryLandBordersCountry": {
        "system_prompt": SYSTEM_PROMPT_BORDERS,
        "subquestions": SUBQUESTIONS_BORDERS,
        "few_shot": FEW_SHOT_BORDERS,
    },
    "personHasCityOfDeath": {
        "system_prompt": SYSTEM_PROMPT_DEATH,
        "subquestions": SUBQUESTIONS_DEATH,
        "few_shot": FEW_SHOT_DEATH,
    },
    "companyTradesAtStockExchange": {
        "system_prompt": SYSTEM_PROMPT_STOCK,
        "subquestions": SUBQUESTIONS_STOCK,
        "few_shot": FEW_SHOT_STOCK,
    },
}

FINAL_INSTRUCTION = {
    "numeric": "Using the subanswers above, commit to a single best estimate now. " + ANSWER_FORMAT_NUMERIC,
    "list": "Using the subanswers above, give the final answer now. " + ANSWER_FORMAT_LIST,
}


def ask_turn(model_id: str, messages: list[dict], user_content: str,
             num_predict: int, label: str) -> str | None:
    """Append one user turn, answer it, and append the answer, keeping
    `messages` a valid alternating chain for the next subquestion.

    Returns the raw response, or None if the model gave nothing usable -- in
    which case the unanswered turn is removed again so no placeholder is left
    for later turns to imitate."""
    messages.append({"role": "user", "content": user_content})

    for temperature in (LTM_TEMPERATURE, EMPTY_RETRY_TEMPERATURE):
        try:
            raw_answer = call_model(model_id, messages, temperature=temperature, num_predict=num_predict)
        except Exception as e:
            print(f"[query error] {model_id} / {label}: {e}")
            break

        subanswer = clean_model_response(raw_answer)
        if subanswer:
            messages.append({"role": "assistant", "content": subanswer})
            return raw_answer

    messages.pop()
    return None


def predict(model_id: str, subject: str, relation: str) -> list[str]:
    config = RELATIONS[relation]
    answer_type = answer_type_for(relation)
    problem = QUESTIONS[relation].format(subject=subject)

    messages = [{
        "role": "system",
        "content": build_system_prompt(model_id, config["system_prompt"]),
    }]

    # The examples and the problem statement ride on the first turn that lands;
    # a dropped turn takes its whole message with it, so they move to the next
    # subquestion rather than being lost.
    preamble_pending = True

    for i, template in enumerate(config["subquestions"], 1):
        subquestion = template.format(subject=subject)
        if preamble_pending:
            content = (
                f"{config['few_shot']}\n"
                f"Problem: {problem}\n\n"
                "Answer the subquestions one at a time, in the order given, using your "
                "earlier answers. Keep each subanswer to one or two sentences, and do "
                "not answer the original question until it is asked.\n\n"
                f"Subquestion {i}: {subquestion}"
            )
        else:
            content = f"Subquestion {i}: {subquestion}"

        answered = ask_turn(model_id, messages, content, SUBANSWER_MAX_TOKENS,
                            f"{subject} ({relation}, subquestion {i})")
        if answered is not None:
            preamble_pending = False

    messages.append({
        "role": "user",
        "content": f"Final question: {problem}\n{FINAL_INSTRUCTION[answer_type]}",
    })

    try:
        raw_answer = call_model(model_id, messages, temperature=LTM_TEMPERATURE)
    except Exception as e:
        print(f"[query error] {model_id} / {subject} ({relation}, final): {e}")
        return []

    return parse_answer(raw_answer, answer_type)


def main():
    args = build_parser(__doc__).parse_args()
    rows = load_rows(args.dataset, RELATIONS, args.limit)
    run_generation(Path(__file__).stem, args.model, rows, args.dataset, predict)


if __name__ == "__main__":
    main()
