import os
from typing import Any

from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4"
GPT4_CONTEXT_WINDOW_TOKENS = 8192
MAX_GENERATION_TOKENS = 512
MAX_PROMPT_TOKENS = GPT4_CONTEXT_WINDOW_TOKENS - MAX_GENERATION_TOKENS


def _get_tokenizer(model: str):
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required to limit GPT-4 prompt size. "
            "Install backend requirements before generating answers."
        ) from exc


def _count_text_tokens(text: str, model: str) -> int:
    tokenizer = _get_tokenizer(model)
    return len(tokenizer.encode(str(text)))


def _count_prompt_tokens(messages: list[dict[str, str]], model: str) -> int:
    tokens = 3

    for message in messages:
        tokens += 3
        tokens += _count_text_tokens(message.get("role", ""), model)
        tokens += _count_text_tokens(message.get("content", ""), model)

    return tokens


def _truncate_text_to_tokens(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0:
        return ""

    tokenizer = _get_tokenizer(model)
    encoded = tokenizer.encode(str(text))

    if len(encoded) <= max_tokens:
        return str(text)

    return tokenizer.decode(encoded[:max_tokens])


def limit_chunks_to_prompt_budget(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> list[dict[str, Any]]:
    """
    Trims retrieved chunk text so the full prompt stays within the token budget.
    """

    limited_chunks = []

    for chunk in retrieved_chunks:
        candidate_chunk = {**chunk, "text": str(chunk.get("text", ""))}
        candidate_messages = build_answer_prompt(
            question=question,
            retrieved_chunks=[*limited_chunks, candidate_chunk],
        )

        if _count_prompt_tokens(candidate_messages, model) <= max_prompt_tokens:
            limited_chunks.append(candidate_chunk)
            continue

        base_messages = build_answer_prompt(
            question=question,
            retrieved_chunks=limited_chunks,
        )
        remaining_tokens = max_prompt_tokens - _count_prompt_tokens(base_messages, model)

        if remaining_tokens <= 0:
            break

        trimmed_chunk = {
            **candidate_chunk,
            "text": _truncate_text_to_tokens(
                candidate_chunk["text"],
                remaining_tokens,
                model,
            ),
        }
        trimmed_messages = build_answer_prompt(
            question=question,
            retrieved_chunks=[*limited_chunks, trimmed_chunk],
        )

        if trimmed_chunk["text"] and _count_prompt_tokens(trimmed_messages, model) <= max_prompt_tokens:
            limited_chunks.append(trimmed_chunk)

        break

    return limited_chunks


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into numbered source blocks for the LLM prompt.
    """

    context_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        source_label = metadata.get("question_id", "unknown")
        chunk_id = chunk.get("chunk_id", metadata.get("chunk_id", "unknown"))

        context_blocks.append(
            f"Source {index} | question_id={source_label} | chunk_id={chunk_id}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_blocks)


def build_answer_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Builds the ChatGPT-4 prompt for grounded financial answer generation.
    """

    context = format_context(retrieved_chunks)

    return [
        {
            "role": "system",
            "content": (
                "You are a financial question-answering assistant. "
                "Answer using only the provided context. "
                "If the answer requires arithmetic, show the calculation briefly. "
                "If the context does not contain enough information, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Return a concise answer and cite the source number(s) used."
            ),
        },
    ]


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Generates a final answer with ChatGPT-4 using retrieved Chroma chunks.
    """

    if not retrieved_chunks:
        return "No relevant context was retrieved from the Chroma index."

    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key is not configured. Set OPENAI_API_KEY to generate an answer."

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    limited_chunks = limit_chunks_to_prompt_budget(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model=model,
    )

    response = client.chat.completions.create(
        model=model,
        messages=build_answer_prompt(question, limited_chunks),
        temperature=0,
        max_tokens=MAX_GENERATION_TOKENS,
    )

    return response.choices[0].message.content.strip()
