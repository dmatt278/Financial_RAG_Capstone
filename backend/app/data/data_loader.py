import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from functools import lru_cache
import ijson
from huggingface_hub import hf_hub_download
from typing import Iterator
from urllib.request import urlopen


TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TUNING_SPLIT = "train_dev"
BASELINE_SPLIT = "test"
FINQA_SPLITS = (TRAIN_SPLIT, DEV_SPLIT, BASELINE_SPLIT)
FINQA_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
)


def get_docfinqa_data_dir() -> Path:
    """Returns the directory used for the combined train/dev file."""

    configured_dir = os.getenv("DOCFINQA_DATA_DIR")
    if configured_dir:
        return Path(configured_dir)
    return Path(__file__).resolve().parents[3] / "data" / "docfinqa"


def get_finqa_data_dir() -> Path:
    """Returns the directory used for the original FinQA evaluation files."""

    configured_dir = os.getenv("FINQA_DATA_DIR")
    if configured_dir:
        return Path(configured_dir)
    return Path(__file__).resolve().parents[3] / "data" / "finqa"


def get_finqa_file_path(split: str) -> str:
    """Downloads or locates an original FinQA split from the official repository."""

    if split not in FINQA_SPLITS:
        raise ValueError(
            f"Invalid FinQA split: {split}. Use 'train', 'dev', or 'test'."
        )

    destination = get_finqa_data_dir() / f"{split}.json"
    if destination.is_file():
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{split}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "wb") as output:
            with urlopen(
                f"{FINQA_DATA_BASE_URL}/{split}.json",
                timeout=60,
            ) as response:
                shutil.copyfileobj(response, output)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return str(destination)


def prepare_finqa_files() -> dict:
    """Downloads and validates the public FinQA splits used for gold evidence."""

    files = {}
    total_records = 0

    for split in FINQA_SPLITS:
        path = Path(get_finqa_file_path(split))
        with open(path, "rb") as source:
            records = sum(1 for _ in ijson.items(source, "item"))

        files[split] = {
            "path": str(path),
            "records": records,
            "bytes": path.stat().st_size,
        }
        total_records += records

    return {
        "data_dir": str(get_finqa_data_dir()),
        "total_records": total_records,
        "files": files,
    }


def _normalize_finqa_question(question: str) -> str:
    return " ".join(str(question).lower().split())


def _normalize_finqa_answer(answer) -> str:
    normalized = (
        str(answer)
        .strip()
        .lower()
        .replace("$", "")
        .replace(",", "")
        .replace("%", "")
    )

    try:
        return f"{float(normalized):.12g}"
    except ValueError:
        return normalized


def _get_finqa_supporting_facts(record: dict) -> list[str]:
    """Gets each gold fact in the form most likely to occur in DocFinQA."""

    qa = record.get("qa", {})
    gold_inds = qa.get("gold_inds") or {}

    if isinstance(gold_inds, dict):
        table = record.get("table") or []
        facts = []

        for evidence_id, value in gold_inds.items():
            evidence_text = str(value).strip() if value is not None else ""

            # FinQA verbalizes table rows in gold_inds. DocFinQA keeps the
            # original row, so use the row selected by the gold table index.
            if str(evidence_id).startswith("table_"):
                _, _, row_index_text = str(evidence_id).partition("_")
                try:
                    row_index = int(row_index_text)
                except ValueError:
                    row_index = -1

                if 0 <= row_index < len(table):
                    raw_row = " ".join(
                        str(cell).strip()
                        for cell in table[row_index]
                        if cell is not None and str(cell).strip()
                    )
                    if raw_row:
                        evidence_text = raw_row

            if evidence_text:
                facts.append(evidence_text)

        return facts

    if isinstance(gold_inds, (list, tuple)):
        return [
            str(value)
            for value in gold_inds
            if value is not None and str(value).strip()
        ]

    raise ValueError(
        f"Unexpected FinQA gold_inds type: {type(gold_inds).__name__}"
    )


@lru_cache(maxsize=2)
def _get_finqa_evidence_lookup(split: str) -> dict[str, list[dict]]:
    """Builds a question lookup for FinQA gold supporting facts."""

    source_splits = ("train", "dev") if split == TUNING_SPLIT else (split,)
    lookup: dict[str, list[dict]] = {}

    for source_split in source_splits:
        path = get_finqa_file_path(source_split)
        with open(path, "rb") as source:
            for record in ijson.items(source, "item"):
                qa = record.get("qa", {})
                question = qa.get("question")
                if not question:
                    continue

                key = _normalize_finqa_question(question)
                lookup.setdefault(key, []).append(
                    {
                        "finqa_id": str(record.get("id", "")),
                        "answer": qa.get("exe_ans", qa.get("answer")),
                        "gold_evidence": _get_finqa_supporting_facts(record),
                    }
                )

    return lookup


def get_finqa_gold_evidence(
    split: str,
    question: str,
    gold_answer,
) -> list[str]:
    """Matches a DocFinQA question to FinQA and returns its gold evidence."""

    lookup = _get_finqa_evidence_lookup(split)
    candidates = lookup.get(_normalize_finqa_question(question), [])

    if not candidates:
        raise ValueError(f"No FinQA evidence match found for question: {question}")

    if len(candidates) == 1:
        return list(candidates[0]["gold_evidence"])

    normalized_answer = _normalize_finqa_answer(gold_answer)
    answer_matches = [
        candidate
        for candidate in candidates
        if _normalize_finqa_answer(candidate["answer"]) == normalized_answer
    ]

    if len(answer_matches) == 1:
        return list(answer_matches[0]["gold_evidence"])

    possible_matches = answer_matches or candidates
    distinct_evidence = {
        tuple(candidate["gold_evidence"])
        for candidate in possible_matches
    }
    if len(distinct_evidence) == 1:
        return list(possible_matches[0]["gold_evidence"])

    match_ids = [candidate["finqa_id"] for candidate in possible_matches]
    raise ValueError(
        f"Ambiguous FinQA evidence match for question '{question}': {match_ids}"
    )


def prepare_train_dev_file(force: bool = False) -> str:
    """Combines DocFinQA train and dev into one streamed JSON file."""

    destination = get_docfinqa_data_dir() / "train_dev.json"
    if destination.is_file() and not force:
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".train_dev.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("[")
            first_record = True

            for source_split in ("train", "dev"):
                source_path = get_docfinqa_file_path(source_split)
                with open(source_path, "rb") as source_file:
                    for raw_example in ijson.items(source_file, "item"):
                        if not first_record:
                            output.write(",")
                        first_record = False
                        json.dump(raw_example, output, ensure_ascii=False)

            output.write("]")

        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return str(destination)


def get_docfinqa_file_path(split: str = TUNING_SPLIT) -> str:
    '''
    Downloads or locates the raw DocFinQA JSON file from Hugging Face.
    '''

    if split == TUNING_SPLIT:
        return prepare_train_dev_file()

    filename_map = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json"
    }

    if split not in filename_map:
        raise ValueError(
            f"Invalid split: {split}. Use 'train', 'dev', 'train_dev', or 'test'."
        )
    
    file_path = hf_hub_download(
        repo_id="kensho/DocFinQA",
        filename=filename_map[split],
        repo_type="dataset",
    )

    return file_path


def convert_docfinqa_fields(raw_example: dict, question_id: int) -> dict:
    '''
    Converts raw DocFinQA fields into cleaner names
    '''

    return {
        "question_id": str(question_id),
        "question": raw_example["Question"],
        "gold_answer": raw_example["Answer"],
        "document_text": raw_example["Context"],
        "document_id": hashlib.sha256(raw_example["Context"].encode("utf-8")).hexdigest(),
        "program": raw_example["Program"],
    }


def load_docfinqa_example(split: str = TUNING_SPLIT, index: int = 0) -> dict:
    """
    Loads one DocFinQA example by index without loading the entire dataset into memory.
    """

    file_path = get_docfinqa_file_path(split)

    with open(file_path, "rb") as file:
        examples = ijson.items(file, "item")

        for current_index, raw_example in enumerate(examples):
            if current_index == index:
                return convert_docfinqa_fields(raw_example, current_index)

    raise IndexError(f"Index {index} is out of range for split '{split}'.")


def load_docfinqa_example_by_question_id(
    split: str = TUNING_SPLIT,
    question_id: str = "1234",
) -> dict:
    """
    Loads one DocFinQA example by question_id without loading the entire dataset.
    """

    for example in iter_docfinqa_examples(split=split):
        if example["question_id"] == str(question_id):
            return example

    raise ValueError(
        f"Question ID {question_id} was not found in split '{split}'."
    )


def iter_docfinqa_examples(
    split: str = TUNING_SPLIT,
    start_index: int = 0,
    limit: int | None = None,
) -> Iterator[dict]:
    """
    Streams DocFinQA examples without loading the full dataset into memory.
    """

    file_path = get_docfinqa_file_path(split)
    yielded = 0

    with open(file_path, "rb") as file:
        examples = ijson.items(file, "item")

        for current_index, raw_example in enumerate(examples):
            if current_index < start_index:
                continue

            if limit is not None and yielded >= limit:
                break

            yielded += 1
            yield convert_docfinqa_fields(raw_example, current_index)


def iter_unique_documents() -> Iterator[dict]:
    """
    Streams each unique DocFinQA document once, deduplicated by document_id
    across the train_dev and test splits.
    """

    seen_document_ids: set[str] = set()

    for split in (TUNING_SPLIT, BASELINE_SPLIT):
        for example in iter_docfinqa_examples(split=split):
            if example["document_id"] in seen_document_ids:
                continue
            seen_document_ids.add(example["document_id"])
            yield {
                "document_id": example["document_id"],
                "document_text": example["document_text"],
            }
