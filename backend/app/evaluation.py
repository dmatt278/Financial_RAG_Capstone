import re
from typing import Dict


def normalize_answer(answer) -> str:
    """
    Normalizes an answer for simple exact-match comparison.

    "$1,200 million" -> "1200"
    """

    if answer is None:
        return ""

    answer = str(answer).strip().lower()

    # Remove common formatting
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")
    answer = answer.replace("%", "")
    answer = answer.replace("million", "")
    answer = answer.replace("millions", "")
    answer = answer.replace("billion", "")
    answer = answer.replace("billions", "")

    # Keep only numbers, decimal points, minus signs, and spaces
    answer = re.sub(r"[^0-9.\-\s]", "", answer)

    answer = answer.strip()

    # Convert 380.0 to 380
    try:
        number = float(answer)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return answer
    
def exact_match(predicted_answer, gold_answer) -> bool:
    """
    Returns True if the normalized predicted answer equals the normalized gold answer.
    """

    normalized_predicted = normalize_answer(predicted_answer)
    normalized_gold = normalize_answer(gold_answer)

    return normalized_predicted == normalized_gold

def evaluate_answer(predicted_answer, gold_answer) -> Dict:
    """
    Compare a predicted answer against the gold answer.
    """

    normalized_predicted = normalize_answer(predicted_answer)
    normalized_gold = normalize_answer(gold_answer)

    is_correct = normalized_predicted == normalized_gold

    return {
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "normalized_predicted": normalized_predicted,
        "normalized_gold": normalized_gold,
        "exact_match": is_correct,
    }

