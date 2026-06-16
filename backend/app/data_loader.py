import ijson
from huggingface_hub import hf_hub_download


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