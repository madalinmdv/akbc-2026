"""Few-shot chain-of-thought prompting.

Each relation gets a system prompt with explicit reasoning steps and a block
of worked examples; the model reasons through the steps in a single turn and
closes with the shared `Answer:` line.
"""

from pathlib import Path

from common.ollama_client import build_system_prompt, call_model
from common.parsing import parse_answer
from common.prompts import ANSWER_FORMAT_LIST, ANSWER_FORMAT_NUMERIC
from common.relations import QUESTIONS, answer_type_for
from common.runner import build_parser, load_rows, run_generation


def render_numeric_few_shot(question_template: str, examples: list[dict]) -> str:
    """Render {"entity", "reasoning", "answer"} examples into the same
    Question/reasoning/Answer blocks the list relations use verbatim."""
    blocks = []
    for example in examples:
        question = question_template.format(subject=example["entity"])
        blocks.append(f"Question: {question}\n{example['reasoning']}\nAnswer: {example['answer']}")
    return "\n\n".join(blocks) + "\n"


# --- hasArea ---------------------------------------------------------------

SYSTEM_PROMPT_AREA = (
    "You are a precise closed-book estimation assistant. For each entity you "
    "reason briefly, then answer. Reason in two steps: "
    "Step 1 - identify what kind of entity it is (lake, island, country, etc.) "
    "and place it in context relative to other entities of similar or "
    "well-known scale. "
    "Step 2 - use that comparison to commit to a single best-estimate numeric "
    "area in square kilometers -- never refuse or say you don't know, even if "
    "you are not certain of the exact figure. "
) + ANSWER_FORMAT_NUMERIC

COT_AREA = [
    {
        "entity": "Moraine Lake in Banff National Park",
        "reasoning": (
            "Moraine Lake is a glacially-fed lake in Banff National Park in the "
            "Canadian Rockies, known for its turquoise water and surrounded by "
            "the Valley of the Ten Peaks. It is a small alpine lake, not a major "
            "freshwater body like the nearby Lake Louise or a large reservoir -- "
            "alpine lakes of this kind typically cover well under a square "
            "kilometer."
        ),
        "answer": "0.5",
    },
    {
        "entity": "Nantucket",
        "reasoning": (
            "Nantucket is an island and county off the coast of Massachusetts, "
            "in the northeastern United States. It is a mid-sized offshore "
            "island, comparable in scale to other well-known New England "
            "islands such as Martha's Vineyard, rather than a small islet or a "
            "large landmass -- such islands are typically on the order of a "
            "hundred or so square kilometers."
        ),
        "answer": "123.8",
    },
    {
        "entity": "Nigeria",
        "reasoning": (
            "Nigeria is a country in West Africa, the most populous country on "
            "the continent, bordering Niger, Chad, Cameroon, and Benin. It is a "
            "large national territory spanning multiple climate zones from "
            "coastal mangroves to Sahelian savanna -- countries of this scale "
            "typically span several hundred thousand square kilometers."
        ),
        "answer": "923768",
    },
]


# --- hasCapacity -----------------------------------------------------------

SYSTEM_PROMPT_CAPACITY = (
    "You are a precise closed-book estimation assistant. For each venue you "
    "reason briefly, then answer. Reason in two steps: "
    "Step 1 - identify the venue's type, tier, and geographic or institutional "
    "context (e.g. small local/municipal ground, collegiate venue, major "
    "professional or national venue), drawing on comparable venues you know. "
    "Step 2 - use that scale to commit to a single best-estimate numeric "
    "capacity -- never refuse or say you don't know, even if you are not "
    "certain of the exact figure. "
) + ANSWER_FORMAT_NUMERIC

COT_CAPACITY = [
    {
        "entity": "Sungui Sports Complex in Incheon",
        "reasoning": (
            "Sungui Sports Complex is a multi-purpose athletic venue in Incheon, "
            "South Korea. Incheon has hosted major multi-sport events including "
            "the 2014 Asian Games and maintains several mid-to-large stadiums. "
            "A city-level multi-purpose sports complex used for athletics and "
            "football typically holds tens of thousands of spectators."
        ),
        "answer": "35000",
    },
    {
        "entity": "John A. Farrell Stadium in Pennsylvania",
        "reasoning": (
            "John A. Farrell Stadium in Pennsylvania is associated with "
            "collegiate and scholastic athletics and football. Venues serving "
            "a college or school district, rather than a major professional "
            "franchise, usually hold a few thousand spectators, sized to the "
            "local community rather than a mass audience."
        ),
        "answer": "7500",
    },
    {
        "entity": "Estadio Revolución Ciudad de Guatemala in Guatemala City",
        "reasoning": (
            "Estadio Revolución is a stadium in Guatemala City, Guatemala. Its "
            "name points to a specific institutional venue rather than the "
            "country's largest national stadium. Secondary or institutional "
            "stadiums in a capital, used for smaller matches, typically have "
            "modest capacities compared with a national flagship stadium."
        ),
        "answer": "5000",
    },
]


# --- countryLandBordersCountry ---------------------------------------------

SYSTEM_PROMPT_BORDERS = (
    "You are a precise geography assistant answering closed-book. "
    "For each country you reason briefly, then answer. "
    "Reason in three steps: "
    "Step 1 - identify the country and its continent/region. "
    "Step 2 - decide whether it can have land borders at all: island nations "
    "and territories surrounded entirely by water have none -- for those the "
    "answer is an empty list. "
    "Step 3 - if it does, name every country it actually shares a land "
    "border with, recalling its full set of neighbors rather than just the "
    "most obvious one or two. "
) + ANSWER_FORMAT_LIST

FEW_SHOT_BORDERS = """Question: Which countries share a land border with New Zealand?
Step 1 - New Zealand is an island country in Oceania, in the southwestern Pacific Ocean, made up of the North Island, South Island, and numerous smaller islands.
Step 2 - As an island nation entirely surrounded by ocean, it has no shared land boundary with any other country.
Step 3 - Since there is no adjoining country, it has no land borders.
Answer: []

Question: Which countries share a land border with Germany?
Step 1 - Germany is a country in Central Europe.
Step 2 - It is a continental country with an extensive land frontier, not an island or enclave, so it does have land borders.
Step 3 - Germany shares land borders with Austria, Belgium, the Czech Republic, Denmark, France, Luxembourg, the Netherlands, Poland, and Switzerland.
Answer: ["Austria", "Belgium", "Czech Republic", "Denmark", "France", "Luxembourg", "Netherlands", "Poland", "Switzerland"]

Question: Which countries share a land border with Sierra Leone?
Step 1 - Sierra Leone is a country on the Atlantic coast of West Africa.
Step 2 - It is a continental West African country, not an island, so it does have land borders.
Step 3 - Sierra Leone shares land borders with Guinea to the north and east, and Liberia to the south and east.
Answer: ["Guinea", "Liberia"]
"""


# --- personHasCityOfDeath --------------------------------------------------

SYSTEM_PROMPT_DEATH = (
    "You identify the city where a person died, answering closed-book. "
    "For each person you reason briefly, then answer. "
    "Reason in three steps: "
    "Step 1 - determine whether the person has died. If they are still living, "
    "or their death cannot be established, the answer is empty. "
    "Step 2 - if they have died, recall where they died specifically. Be careful "
    "NOT to confuse the city they are most FAMOUS for, or lived or worked in, "
    "with the city where they actually died -- these are frequently different. "
    "Step 3 - give the single city of death (city granularity, not country or "
    "region). If you are genuinely unsure of the city, the answer is empty "
    "rather than a guess. "
) + ANSWER_FORMAT_LIST

FEW_SHOT_DEATH = """Question: In which city did Sigmund Freud die?
Step 1 - Sigmund Freud, the founder of psychoanalysis, died in 1939; he is deceased.
Step 2 - He is overwhelmingly associated with Vienna, where he lived and worked for most of his life. However, he fled Vienna in 1938 after the Nazi annexation of Austria and died the following year in London. The famous city is not the city of death.
Step 3 - The city of death is London.
Answer: ["London"]

Question: In which city did Uwe Timm die?
Step 1 - Uwe Timm, the German author, is still living as of current records.
Step 2 - Since he has not died, there is no city of death to identify.
Step 3 - For a living person the answer is empty.
Answer: []

Question: In which city did Albert Einstein die?
Step 1 - Albert Einstein, the theoretical physicist, died in 1955; he is deceased.
Step 2 - He is famous for his work in Germany, Zurich, and Berlin, but he spent his final years in the United States and died in Princeton, New Jersey. The city of death differs from the cities he is famous for.
Step 3 - The city of death is Princeton.
Answer: ["Princeton"]

Question: In which city did Henk van Kerkwijk die?
Step 1 - I cannot confidently verify whether Henk van Kerkwijk has died, nor locate reliable records of a death city.
Step 2 - Since his death cannot be established, a specific city of death cannot be identified.
Step 3 - When the death is unverifiable, the answer is empty rather than a guess.
Answer: []

Question: In which city did Milan Lasica die?
Step 1 - Milan Lasica, the Slovak actor and writer, has died.
Step 2 - He lived and worked in Bratislava, and in this case he also died in Bratislava; here the death city and the lived city coincide.
Step 3 - The city of death is Bratislava.
Answer: ["Bratislava"]

Question: In which city did Lasse Åberg die?
Step 1 - Lasse Åberg, the Swedish actor and director, is still living as of current records.
Step 2 - Since he has not died, there is no city of death to identify.
Step 3 - For a living person the answer is empty.
Answer: []
"""


# --- companyTradesAtStockExchange ------------------------------------------

SYSTEM_PROMPT_STOCK = (
    "You are a precise financial-facts assistant answering closed-book. "
    "For each company you reason briefly, then answer. "
    "Reason in three steps: "
    "Step 1 - identify the company (country, sector, ownership structure). "
    "Step 2 - decide whether it is publicly listed at all: many companies are "
    "privately held, are wholly-owned subsidiaries not separately listed, or "
    "have been delisted -- for those the answer is an empty list. "
    "Step 3 - if it is listed, name every exchange where its own shares trade, "
    "including secondary, dual, and ADR/GDR listings. "
) + ANSWER_FORMAT_LIST

FEW_SHOT_STOCK = """Question: On which stock exchanges does Bangladesh Cement Manufacturers Association trade?
Step 1 - The Bangladesh Cement Manufacturers Association (BCMA) is a trade/industry association representing cement producers in Bangladesh, not a commercial corporation or operating company.
Step 2 - As a membership-based industry body, it does not issue equity shares to the public and is therefore not publicly listed on any exchange.
Step 3 - Since it is not a publicly traded entity, it has no stock exchange listings.
Answer: []

Question: On which stock exchanges does West Japan Railway Company trade?
Step 1 - West Japan Railway Company (JR West) is a Japanese transportation company operating passenger and freight railways across the Kansai, Chugoku, and Shikoku regions. It was formed during the privatization of the former state-owned Japanese National Railways and operates as an independent publicly held corporation.
Step 2 - The company is publicly listed; it conducted its initial public offering in October 1991 and remains actively traded on the open market.
Step 3 - Its shares trade exclusively on the Tokyo Stock Exchange (Prime Market, ticker 9021). It does not maintain active secondary domestic cross-listings or any ADR/GDR programs on foreign exchanges.
Answer: ["Tokyo Stock Exchange"]

Question: On which stock exchanges does Energen trade?
Step 1 - Energen Corporation is a United States-based electric utility company headquartered in Birmingham, Alabama, operating primarily in the Southeastern U.S. It is structured as a publicly traded corporation.
Step 2 - The company is publicly listed and actively trades on the open market; it is not privately held, delisted, or a non-listed subsidiary.
Step 3 - Energen's common stock trades exclusively on the New York Stock Exchange under the ticker symbol EGN. It has no secondary, dual, or ADR/GDR listings on other exchanges.
Answer: ["New York Stock Exchange"]
"""


RELATIONS = {
    "hasArea": {
        "system_prompt": SYSTEM_PROMPT_AREA,
        "few_shot": render_numeric_few_shot(QUESTIONS["hasArea"], COT_AREA),
    },
    "hasCapacity": {
        "system_prompt": SYSTEM_PROMPT_CAPACITY,
        "few_shot": render_numeric_few_shot(QUESTIONS["hasCapacity"], COT_CAPACITY),
    },
    "countryLandBordersCountry": {
        "system_prompt": SYSTEM_PROMPT_BORDERS,
        "few_shot": FEW_SHOT_BORDERS,
    },
    "personHasCityOfDeath": {
        "system_prompt": SYSTEM_PROMPT_DEATH,
        "few_shot": FEW_SHOT_DEATH,
    },
    "companyTradesAtStockExchange": {
        "system_prompt": SYSTEM_PROMPT_STOCK,
        "few_shot": FEW_SHOT_STOCK,
    },
}


def predict(model_id: str, subject: str, relation: str) -> list[str]:
    config = RELATIONS[relation]
    question = QUESTIONS[relation].format(subject=subject)
    messages = [
        {"role": "system", "content": build_system_prompt(model_id, config["system_prompt"])},
        {"role": "user", "content": f"{config['few_shot']}\nQuestion: {question}\n"},
    ]

    try:
        raw_answer = call_model(model_id, messages)
    except Exception as e:
        print(f"[query error] {model_id} / {subject} ({relation}): {e}")
        return []

    return parse_answer(raw_answer, answer_type_for(relation))


def main():
    args = build_parser(__doc__).parse_args()
    rows = load_rows(args.dataset, RELATIONS, args.limit)
    run_generation(Path(__file__).stem, args.model, rows, args.dataset, predict)


if __name__ == "__main__":
    main()
