import ijson
from huggingface_hub import hf_hub_download
from typing import Iterator


def get_docfinqa_file_path(split: str = "train") -> str:
    '''
    Downloads or locates the raw DocFinQA JSON file from Hugging Face.
    '''

    filename_map = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json"
    }

    if split not in filename_map:
        raise ValueError(f"Invalid split: {split}. Use 'train', 'dev', or 'test'.")
    
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
        "program": raw_example["Program"],
    }


def load_docfinqa_example(split: str = "train", index: int = 0) -> dict:
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
    split: str = "train",
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
    split: str = "train",
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
