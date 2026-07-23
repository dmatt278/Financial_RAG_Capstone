import hashlib
import json
import os
from pathlib import Path
import tempfile
import ijson
from huggingface_hub import hf_hub_download
from typing import Iterator


TUNING_SPLIT = "train_dev"
BASELINE_SPLIT = "test"


def get_docfinqa_data_dir() -> Path:
    """Returns the directory used for the combined train/dev file."""

    configured_dir = os.getenv("DOCFINQA_DATA_DIR")
    if configured_dir:
        return Path(configured_dir)
    return Path(__file__).resolve().parents[3] / "data" / "docfinqa"


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
