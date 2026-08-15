"""The relations of the task: how each one is answered and how it is asked."""

AWARD_RELATION = "awardWonBy"

# "numeric" relations expect a single number, "list" relations a set of names
# (possibly empty). This drives which system prompt and which parser is used.
ANSWER_TYPES = {
    "hasArea": "numeric",
    "hasCapacity": "numeric",
    "countryLandBordersCountry": "list",
    "personHasCityOfDeath": "list",
    "companyTradesAtStockExchange": "list",
    AWARD_RELATION: "list",
}

# Canonical natural-language phrasing of each relation, shared by the
# reasoning strategies so they are compared on identical questions.
QUESTIONS = {
    "hasArea": "What is the area of {subject} in square kilometers?",
    "hasCapacity": "What is the total capacity of {subject}?",
    "countryLandBordersCountry": "Which countries share a land border with {subject}?",
    "personHasCityOfDeath": "In which city did {subject} die?",
    "companyTradesAtStockExchange": "On which stock exchanges does {subject} trade?",
}


def answer_type_for(relation: str) -> str:
    return ANSWER_TYPES.get(relation, "list")
