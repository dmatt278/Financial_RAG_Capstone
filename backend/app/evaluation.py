import re
from typing import Dict


def normalize_text(text) -> str:
    """
    Normalizes an answer for simple exact-match comparison.

    "$1,200 million" -> "1200"
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Remove common formatting
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.replace("million", "")
    text = text.replace("millions", "")
    text = text.replace("billion", "")
    text = text.replace("billions", "")

    # Keep only numbers, decimal points, minus signs, and spaces
    text = re.sub(r"[^0-9.\-\s]", "", text)

    text = text.strip()

    # Convert 380.0 to 380
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return text
    
def normalize_numeric_string(s: str) -> float | None:
    """Convert a raw answer string into a float, handling common financial formatting."""
    if s is None:
        return None
    
    s = str(s).strip()
    
    # Handle negative numbers in parentheses, e.g. "(1,200)" -> "-1200"
    is_negative = s.startswith('(') and s.endswith(')')
    if is_negative:
        s = s[1:-1]
    
    # Strip currency symbols, commas, whitespace
    s = re.sub(r'[$,\s]', '', s)
    
    # Handle percentage
    is_percentage = s.endswith('%')
    if is_percentage:
        s = s[:-1]
    
    # Handle magnitude suffixes (B/M/K, case-insensitive)
    multiplier = 1
    magnitude_map = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}
    if s and s[-1].lower() in magnitude_map:
        multiplier = magnitude_map[s[-1].lower()]
        s = s[:-1]
    
    # Handle "billion"/"million" spelled out
    words_map = {'billion': 1e9, 'million': 1e6, 'thousand': 1e3, 'trillion': 1e12}
    for word, mult in words_map.items():
        if word in s.lower():
            s = re.sub(word, '', s, flags=re.IGNORECASE).strip()
            multiplier = mult
            break
    
    try:
        value = float(s)
    except ValueError:
        return None
    
    value *= multiplier
    if is_percentage:
        value /= 100
    if is_negative:
        value = -abs(value)
    
    return value
    
    
def exact_match(predicted_answer, normalized_gold) -> bool:
    """
    Returns True if the normalized predicted answer equals the normalized gold answer.
    """

    normalized_predicted = normalize_text(predicted_answer)
    normalized_gold = normalize_text(normalized_gold)

    return normalized_predicted == normalized_gold


def extract_program_numbers(program):
    """Supporting facts = the document values the program operates on."""
    cleaned = re.sub(r"#\d+", " ", program)        # drop refs like #0, #1
    cleaned = re.sub(r"const_\S+", " ", cleaned)   # drop constants like const_100
    cleaned = cleaned.replace(",", "")
    return set(re.findall(r"-?\d+\.?\d*", cleaned))


def _chunk_has_number(chunk_text, number):
    haystack = chunk_text.replace(",", "")          # "5,214" -> "5214"
    return re.search(rf"(?<!\d){re.escape(number)}(?!\d)", haystack) is not None


def docfinqa_is_relevant(chunk_text, program):
    nums = extract_program_numbers(program)
    return any(_chunk_has_number(chunk_text, n) for n in nums)


def _tokens(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def financebench_is_relevant(chunk_text, evidence, n=6, threshold=0.5):
    """Relevant if a chunk shares enough n-grams with any evidence quote."""
    ct = _tokens(chunk_text)
    chunk_ngrams = set(zip(*[ct[i:] for i in range(n)])) if len(ct) >= n else set()
    for quote in evidence:
        q = _tokens(quote)
        if len(q) < n:
            if q and all(t in ct for t in q):
                return True
            continue
        q_ngrams = set(zip(*[q[i:] for i in range(n)]))
        if q_ngrams and len(q_ngrams & chunk_ngrams) / len(q_ngrams) >= threshold:
            return True
    return False


def precision_at_k(chunks, k) -> float:
    '''
    Precision@k = relevant retrieved chunks in top k / k
    '''
    count = 0
    for data in chunks[:k]:
        if data["is_relevant"]:
            count += 1

    return count / k, count

def recall_at_k(chunks, k):
    '''
    Relevant chunks in top k / total relevant chunks retrieved.
    Returns None if no chunk was labeled relevant (a labeling gap, not a miss).
    '''
    total_relevant = sum(1 for c in chunks if c["is_relevant"])
    if total_relevant == 0:
        return None
    relevant_in_top_k = sum(1 for c in chunks[:k] if c["is_relevant"])
    return relevant_in_top_k / total_relevant

def reciprocal_rank(chunks) -> float:
    '''
    Reciprocal Rank = 1 / rank of first relevant chunk
    '''
    rank = 1
    for data in chunks:
        if data["is_relevant"]:
            return 1 / rank
        else:
            rank += 1
    return 0


def evaluate_retrieval(chunks, k, *, program=None, evidence=None):
    for c in chunks:
        c["is_relevant"] = (docfinqa_is_relevant(c["text"], program)
                            if program is not None
                            else financebench_is_relevant(c["text"], evidence))
    precision, rel_count = precision_at_k(chunks, k)
    return {
        "precision_at_k": precision,
        "recall_at_k": recall_at_k(chunks, k),
        "reciprocal_rank": reciprocal_rank(chunks),
        "relevant_retrieved_count": rel_count,
        "top_k": k,
    }



def evaluate_docfinqa_answer(generated_answer, gold_answer):
    normalized_generated = normalize_numeric_string(generated_answer)
    normalized_gold = normalize_numeric_string(gold_answer)

    if normalized_generated is None or normalized_gold is None:
        return False

    if normalized_gold == 0:
        return abs(normalized_generated - normalized_gold) <= 1e-6

    return abs(normalized_generated - normalized_gold) / abs(normalized_gold) <= 0.01