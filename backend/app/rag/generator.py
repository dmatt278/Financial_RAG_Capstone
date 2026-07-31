import os
from typing import Any, Callable


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
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
    prompt_builder: Callable[
        [str, list[dict[str, Any]]],
        list[dict[str, str]],
    ] | None = None,
) -> list[dict[str, Any]]:
    """
    Trims retrieved chunk text so the full prompt stays within the token budget.
    """

    prompt_builder = prompt_builder or build_answer_prompt
    limited_chunks = []

    for chunk in retrieved_chunks:
        candidate_chunk = {**chunk, "text": str(chunk.get("text", ""))}
        candidate_messages = prompt_builder(
            question=question,
            retrieved_chunks=[*limited_chunks, candidate_chunk],
        )

        if _count_prompt_tokens(candidate_messages, model) <= max_prompt_tokens:
            limited_chunks.append(candidate_chunk)
            continue

        tokenizer = _get_tokenizer(model)
        encoded_text = tokenizer.encode(candidate_chunk["text"])
        lowest_length = 0
        highest_length = len(encoded_text)
        fitted_text = ""

        # Find the longest prefix that fits after the prompt builder adds
        # source labels and other formatting around the chunk.
        while lowest_length <= highest_length:
            candidate_length = (lowest_length + highest_length) // 2
            candidate_text = tokenizer.decode(
                encoded_text[:candidate_length]
            )
            trimmed_chunk = {
                **candidate_chunk,
                "text": candidate_text,
            }
            trimmed_messages = prompt_builder(
                question=question,
                retrieved_chunks=[*limited_chunks, trimmed_chunk],
            )

            if (
                _count_prompt_tokens(trimmed_messages, model)
                <= max_prompt_tokens
            ):
                fitted_text = candidate_text
                lowest_length = candidate_length + 1
            else:
                highest_length = candidate_length - 1

        if fitted_text:
            limited_chunks.append(
                {
                    **candidate_chunk,
                    "text": fitted_text,
                }
            )

        break

    return limited_chunks


def _chunk_identifier(chunk: dict[str, Any]) -> Any:
    metadata = chunk.get("metadata", {})
    return chunk.get(
        "id",
        chunk.get("chunk_id", metadata.get("chunk_id")),
    )


def _generation_context_metrics(
    source_chunks: list[dict[str, Any]],
    generation_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describes how much retrieved context reached the generation prompt."""

    prompt_was_truncated = (
        len(source_chunks) != len(generation_chunks)
        or any(
            str(source.get("text", ""))
            != str(generated.get("text", ""))
            for source, generated in zip(source_chunks, generation_chunks)
        )
    )
    return {
        "source_context_chunk_count": len(source_chunks),
        "generation_context_chunk_count": len(generation_chunks),
        "generation_context_chunk_ids": [
            _chunk_identifier(chunk)
            for chunk in generation_chunks
        ],
        "source_context_character_count": sum(
            len(str(chunk.get("text", "")))
            for chunk in source_chunks
        ),
        "generation_context_character_count": sum(
            len(str(chunk.get("text", "")))
            for chunk in generation_chunks
        ),
        "prompt_was_truncated": prompt_was_truncated,
    }


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
                "Give a concise calculation and cite the source number(s) used. "
                "End with a separate line exactly in this format: "
                "FINAL_ANSWER: <number, percentage, yes, or no>. "
                "Preserve the answer's unit, including a percent sign when applicable."
            ),
        },
    ]


def build_no_context_answer_prompt(question: str) -> list[dict[str, str]]:
    """Builds the closed-book baseline prompt without retrieved context."""

    return [
        {
            "role": "system",
            "content": (
                "You are a financial question-answering assistant. "
                "Answer using only your internal knowledge and reasoning. "
                "You have not been given the source financial document. "
                "Do not browse, use external tools, or assume access to retrieved "
                "context. If the available information is insufficient, say so. "
                "If the question contains enough information to calculate the "
                "answer, perform the calculation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "End with a separate line exactly in this format: "
                "FINAL_ANSWER: <number, percentage, yes, or no>. "
                "Preserve the answer's unit, including a percent sign when applicable."
            ),
        },
    ]


def generate_no_context_answer(question: str) -> str:
    """Generates a closed-book answer using the question alone."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required to generate a no-context answer."
        )

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=build_no_context_answer_prompt(question),
        temperature=0,
        max_tokens=MAX_GENERATION_TOKENS,
    )

    return response.choices[0].message.content.strip()


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    generation_context_metrics: dict[str, Any] | None = None,
) -> str:
    """
    Generates a final answer with ChatGPT-4 using retrieved Chroma chunks.
    """

    if not retrieved_chunks:
        if generation_context_metrics is not None:
            generation_context_metrics.update(
                _generation_context_metrics([], [])
            )
        return "No relevant context was retrieved from the Chroma index."

    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key is not configured. Set OPENAI_API_KEY to generate an answer."

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    limited_chunks = limit_chunks_to_prompt_budget(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model=model,
    )
    if generation_context_metrics is not None:
        generation_context_metrics.update(
            _generation_context_metrics(
                source_chunks=retrieved_chunks,
                generation_chunks=limited_chunks,
            )
        )

    response = client.chat.completions.create(
        model=model,
        messages=build_answer_prompt(question, limited_chunks),
        temperature=0,
        max_tokens=MAX_GENERATION_TOKENS,
    )

    return response.choices[0].message.content.strip()
