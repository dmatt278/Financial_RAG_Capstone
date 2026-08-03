import math
import re


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


DOCFINQA_NGRAM_SIZE = 4
FINAL_ANSWER_PATTERN = re.compile(
    r"^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[index:] for index in range(n)]))


def _tokens(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def _docfinqa_evidence_overlap_from_features(
    chunk_token_set,
    chunk_ngrams,
    evidence_tokens,
    evidence_ngrams,
    n=DOCFINQA_NGRAM_SIZE,
):
    """Calculates evidence overlap from precomputed token features."""

    if not evidence_tokens:
        return 0.0

    if len(evidence_tokens) < n:
        matched_tokens = sum(
            1
            for token in evidence_tokens
            if token in chunk_token_set
        )
        return matched_tokens / len(evidence_tokens)

    if not evidence_ngrams:
        return 0.0

    return len(chunk_ngrams & evidence_ngrams) / len(evidence_ngrams)


def docfinqa_evidence_overlap(
    chunk_text,
    gold_evidence,
    n=DOCFINQA_NGRAM_SIZE,
):
    """Measures how much of one FinQA supporting fact appears in a chunk."""

    chunk_tokens = _tokens(chunk_text)
    evidence_tokens = _tokens(gold_evidence)

    chunk_ngrams = _ngrams(chunk_tokens, n)
    evidence_ngrams = _ngrams(evidence_tokens, n)
    return _docfinqa_evidence_overlap_from_features(
        set(chunk_tokens),
        chunk_ngrams,
        evidence_tokens,
        evidence_ngrams,
        n=n,
    )


def align_docfinqa_evidence(gold_evidence, all_question_chunks):
    """Aligns each FinQA supporting fact to its best DocFinQA chunk or chunks."""

    if gold_evidence is None:
        raise ValueError("gold_evidence is required for DocFinQA evaluation.")
    if not all_question_chunks:
        raise ValueError(
            "No candidate chunks matched this document, strategy, and chunk size."
        )

    unique_evidence = []
    seen_evidence = set()
    for evidence_text in gold_evidence:
        evidence_tokens = _tokens(evidence_text)
        normalized = " ".join(evidence_tokens)
        if normalized and normalized not in seen_evidence:
            seen_evidence.add(normalized)
            unique_evidence.append(
                {
                    "text": str(evidence_text),
                    "tokens": evidence_tokens,
                    "ngrams": _ngrams(
                        evidence_tokens,
                        DOCFINQA_NGRAM_SIZE,
                    ),
                }
            )

    if not unique_evidence:
        raise ValueError("FinQA returned no usable gold supporting evidence.")

    chunk_features = []
    for chunk in all_question_chunks:
        chunk_tokens = _tokens(chunk["text"])
        chunk_features.append(
            {
                "id": chunk["id"],
                "token_set": set(chunk_tokens),
                "ngrams": _ngrams(
                    chunk_tokens,
                    DOCFINQA_NGRAM_SIZE,
                ),
            }
        )

    alignments = []
    for evidence in unique_evidence:
        scored_chunks = [
            (
                _docfinqa_evidence_overlap_from_features(
                    chunk["token_set"],
                    chunk["ngrams"],
                    evidence["tokens"],
                    evidence["ngrams"],
                ),
                chunk["id"],
            )
            for chunk in chunk_features
        ]
        best_score = max(
            (score for score, chunk_id in scored_chunks),
            default=0.0,
        )
        target_chunk_ids = sorted({
            chunk_id
            for score, chunk_id in scored_chunks
            if best_score > 0 and score == best_score
        })

        alignments.append(
            {
                "evidence": evidence["text"],
                "score": best_score,
                "target_chunk_ids": target_chunk_ids,
            }
        )

    return alignments


def docfinqa_is_relevant(chunk_id, relevant_chunk_ids):
    """Returns whether a retrieved chunk is aligned to FinQA gold evidence."""

    return chunk_id in relevant_chunk_ids


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


def recall_at_k(chunks, k, total_relevant=None):
    '''
    Relevant chunks in top k / total relevant chunks in the candidate set.
    '''
    if total_relevant is None:
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


def evaluate_retrieval(
    chunks,
    k,
    *,
    evidence=None,
    gold_evidence=None,
    all_question_chunks=None,
    evidence_alignment=None,
):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if gold_evidence is not None or evidence_alignment is not None:
        alignments = (
            evidence_alignment
            if evidence_alignment is not None
            else align_docfinqa_evidence(
                gold_evidence=gold_evidence,
                all_question_chunks=all_question_chunks,
            )
        )
        relevant_chunk_ids = {
            chunk_id
            for alignment in alignments
            for chunk_id in alignment["target_chunk_ids"]
        }

        for chunk in chunks:
            chunk["is_relevant"] = docfinqa_is_relevant(
                chunk["id"],
                relevant_chunk_ids,
            )

        retrieved_ids = {
            chunk["id"]
            for chunk in chunks[:k]
        }
        covered_evidence = sum(
            1
            for alignment in alignments
            if retrieved_ids & set(alignment["target_chunk_ids"])
        )
        gold_evidence_count = len(alignments)
        aligned_evidence_count = sum(
            1
            for alignment in alignments
            if alignment["target_chunk_ids"]
        )
        evidence_recall = (
            covered_evidence / gold_evidence_count
            if gold_evidence_count
            else None
        )
        hit_rate = (
            float(covered_evidence > 0)
            if gold_evidence_count
            else None
        )
        all_evidence_hit = (
            float(covered_evidence == gold_evidence_count)
            if gold_evidence_count
            else None
        )
        alignment_scores = [
            alignment["score"]
            for alignment in alignments
        ]
        total_relevant = len(relevant_chunk_ids)
    else:
        if evidence is None:
            raise ValueError(
                "Gold evidence is required to evaluate retrieval relevance."
            )

        for chunk in chunks:
            chunk["is_relevant"] = financebench_is_relevant(
                chunk["text"],
                evidence,
            )

        gold_evidence_count = len(evidence)
        aligned_evidence_count = None
        covered_evidence = None
        evidence_recall = None
        hit_rate = None
        all_evidence_hit = None
        alignment_scores = []
        total_relevant = None

    precision, rel_count = precision_at_k(chunks, k)
    return {
        "relevance_method": (
            "finqa_gold_evidence_fourgram"
            if evidence_alignment is not None or gold_evidence is not None
            else "financebench_evidence_overlap"
        ),
        "precision_at_k": precision,
        "recall_at_k": recall_at_k(
            chunks,
            k,
            total_relevant=total_relevant,
        ),
        "reciprocal_rank": reciprocal_rank(chunks),
        "relevant_retrieved_count": rel_count,
        "retrieved_count_at_k": len(chunks[:k]),
        "total_relevant_chunks": total_relevant,
        "hit_rate_at_k": hit_rate,
        "all_evidence_hit_at_k": all_evidence_hit,
        "evidence_recall_at_k": evidence_recall,
        "gold_evidence_count": gold_evidence_count,
        "aligned_evidence_count": aligned_evidence_count,
        "retrieved_evidence_count": covered_evidence,
        "average_alignment_score": (
            sum(alignment_scores) / len(alignment_scores)
            if alignment_scores
            else None
        ),
        "minimum_alignment_score": (
            min(alignment_scores)
            if alignment_scores
            else None
        ),
        "top_k": k,
    }


def extract_final_answer(generated_answer) -> str:
    """Extracts the explicit final answer while retaining bare-answer support."""

    answer_text = "" if generated_answer is None else str(generated_answer).strip()
    matches = FINAL_ANSWER_PATTERN.findall(answer_text)
    if matches:
        return matches[-1].strip().rstrip(".")
    return answer_text


DOCFINQA_MAGNITUDE_SUFFIXES = {
    "thousand": "thousand",
    "million": "million",
    "billion": "billion",
    "trillion": "trillion",
    "k": "thousand",
    "m": "million",
    "b": "billion",
    "t": "trillion",
}
DOCFINQA_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def _docfinqa_numeric_metadata(answer) -> dict | None:
    """
    Parses a final number while retaining displayed precision.

    DocFinQA inconsistently includes magnitude words such as "million" in its
    answer labels, so those words are treated as formatting rather than scaling.
    """

    text = extract_final_answer(answer).strip().lower().rstrip(".")
    if not text:
        return None

    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()

    text = text.replace("$", "").replace(",", "").strip()
    is_percentage = text.endswith("%")
    if is_percentage:
        text = text[:-1].strip()

    magnitude_suffix = None
    for suffix, canonical_suffix in DOCFINQA_MAGNITUDE_SUFFIXES.items():
        if text.endswith(suffix):
            numeric_text = text[:-len(suffix)].strip()
            if DOCFINQA_NUMBER_PATTERN.fullmatch(numeric_text):
                text = numeric_text
                magnitude_suffix = canonical_suffix
                break

    if not DOCFINQA_NUMBER_PATTERN.fullmatch(text):
        return None

    decimal_places = (
        len(text.partition(".")[2])
        if "." in text
        else 0
    )
    display_value = float(text)
    if negative_parentheses:
        display_value = -abs(display_value)

    normalized_value = display_value
    display_resolution = 10 ** -decimal_places
    if is_percentage:
        normalized_value /= 100
        display_resolution /= 100

    return {
        "normalized_value": normalized_value,
        "display_value": display_value,
        "decimal_places": decimal_places,
        "display_resolution": display_resolution,
        "is_percentage": is_percentage,
        "magnitude_suffix": magnitude_suffix,
    }


def normalize_docfinqa_answer(answer):
    """Normalizes numeric and yes/no DocFinQA answers for comparison."""

    answer_text = extract_final_answer(answer)
    normalized_text = answer_text.strip().lower().rstrip(".")
    if normalized_text in {"yes", "no"}:
        return normalized_text
    numeric_metadata = _docfinqa_numeric_metadata(answer_text)
    if numeric_metadata is None:
        return None
    return numeric_metadata["normalized_value"]


def evaluate_docfinqa_answer_metrics(generated_answer, gold_answer) -> dict:
    """Calculates benchmark-style final-answer metrics."""

    normalized_generated = normalize_docfinqa_answer(generated_answer)
    normalized_gold = normalize_docfinqa_answer(gold_answer)
    generated_numeric_metadata = _docfinqa_numeric_metadata(generated_answer)
    gold_numeric_metadata = _docfinqa_numeric_metadata(gold_answer)
    parse_succeeded = normalized_generated is not None
    comparable = (
        normalized_generated is not None
        and normalized_gold is not None
        and type(normalized_generated) is type(normalized_gold)
    )

    absolute_error = None
    relative_error = None
    within_one_percent = False
    exact_normalized_match = False
    gold_display_resolution = None
    gold_display_decimal_places = None
    explicit_magnitude_conflict = False

    if comparable and isinstance(normalized_gold, float):
        absolute_error = abs(normalized_generated - normalized_gold)
        generated_magnitude = generated_numeric_metadata["magnitude_suffix"]
        gold_magnitude = gold_numeric_metadata["magnitude_suffix"]
        explicit_magnitude_conflict = (
            generated_magnitude is not None
            and gold_magnitude is not None
            and generated_magnitude != gold_magnitude
        )
        exact_normalized_match = math.isclose(
            normalized_generated,
            normalized_gold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) and not explicit_magnitude_conflict
        if explicit_magnitude_conflict:
            within_one_percent = False
        elif normalized_gold == 0:
            within_one_percent = absolute_error <= 1e-6
        else:
            relative_error = absolute_error / abs(normalized_gold)
            within_one_percent = relative_error <= 0.01

        gold_display_resolution = gold_numeric_metadata[
            "display_resolution"
        ]
        gold_display_decimal_places = gold_numeric_metadata[
            "decimal_places"
        ]
        rounding_tolerance = gold_display_resolution / 2
        is_correct = not explicit_magnitude_conflict and (
            absolute_error
            <= rounding_tolerance
            + max(1e-12, gold_display_resolution * 1e-9)
        )
    elif comparable:
        is_correct = normalized_generated == normalized_gold
        exact_normalized_match = is_correct
    else:
        is_correct = False

    return {
        "docfinqa_answer_correct": is_correct,
        "answer_parse_succeeded": parse_succeeded,
        "normalized_generated_answer": normalized_generated,
        "normalized_gold_answer": normalized_gold,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "within_one_percent": within_one_percent,
        "exact_normalized_match": exact_normalized_match,
        "rounded_to_gold_precision_match": is_correct,
        "explicit_magnitude_conflict": explicit_magnitude_conflict,
        "gold_display_decimal_places": gold_display_decimal_places,
        "gold_display_resolution": gold_display_resolution,
    }


def evaluate_docfinqa_answer(generated_answer, gold_answer):
    return evaluate_docfinqa_answer_metrics(
        generated_answer=generated_answer,
        gold_answer=gold_answer,
    )["docfinqa_answer_correct"]
