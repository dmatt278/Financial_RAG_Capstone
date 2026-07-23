from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ChunkerConfig:
    """
    Stores tokenizer and regex settings shared by all chunking strategies.
    """

    tokenizer: object | None
    table_line: re.Pattern
    header_line: re.Pattern
    part_re: re.Pattern
    item_re: re.Pattern


def _get_tokenizer():
    """
    Gets the tokenizer used by both token counting and LlamaIndex splitters.
    """

    try:
        import tiktoken

        return tiktoken.encoding_for_model("gpt-4o").encode
    except Exception:
        return None


def _build_config():
    """
    Builds the shared configuration used throughout the chunker.
    """

    return ChunkerConfig(
        tokenizer=_get_tokenizer(),
        table_line=re.compile(r'^\s*\|.*\|\s*$'),
        header_line=re.compile(r'^\s*#{1,6}\s+(.+?)\s*$'),
        part_re=re.compile(r'^\s*part\s+([ivxlcdm]+)\b', re.I),
        item_re=re.compile(r'^\s*item\s+(\d+[a-c]?)\b', re.I),
    )


_CONFIG = _build_config()


def count_tokens(s):
    """
    Counts tokens with the shared tokenizer, or estimates if it is unavailable.
    """

    if _CONFIG.tokenizer is None:
        return max(1, len(s) // 4)

    return len(_CONFIG.tokenizer(s))


def make_chunk(chunk_id, text, strategy, chunk_size,
               part=None, item=None, header=None, is_table=False,
               start=None, end=None):
    """
    Builds the standard chunk dictionary used by all chunking strategies.
    """

    return {
        "chunk_id": chunk_id,
        "text": text,
        "tokens": count_tokens(text),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "part": part or "",
        "item": item or "",
        "header": header or "",
        "is_table": is_table,
        "start": -1 if start is None else start,
        "end": -1 if end is None else end,
    }


def _sentence_splitter(size):
    """
    Creates the sentence splitter shared by sentence and section chunking.
    """

    return SentenceSplitter(chunk_size=size, chunk_overlap=0, tokenizer=_CONFIG.tokenizer)


def fixed_size_chunks(text, size=256):
    """
    Splits text into fixed token windows without respecting sentence boundaries.
    """

    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=0, tokenizer=_CONFIG.tokenizer)
    return [make_chunk(i, c, "fixed", size)
            for i, c in enumerate(splitter.split_text(text))]


def sentence_aware_chunks(text, size=256):
    """
    Splits text into chunks that try to preserve sentence boundaries.
    """

    splitter = _sentence_splitter(size)
    return [make_chunk(i, c, "sentence", size)
            for i, c in enumerate(splitter.split_text(text))]


def segment(text):
    """
    Splits text into ordered header, table, and prose segments.
    """

    lines = text.splitlines(keepends=True)
    segs, pos, i = [], 0, 0
    while i < len(lines):
        start = pos
        if lines[i].strip().startswith('#'):
            segs.append(('header', lines[i].strip(), start))
            pos += len(lines[i]); i += 1
        elif _CONFIG.table_line.match(lines[i]):
            buf = []
            while i < len(lines) and _CONFIG.table_line.match(lines[i]):
                buf.append(lines[i]); pos += len(lines[i]); i += 1
            segs.append(('table', ''.join(buf).strip(), start))
        else:
            buf = []
            while (i < len(lines) and not lines[i].strip().startswith('#')
                   and not _CONFIG.table_line.match(lines[i])):
                buf.append(lines[i]); pos += len(lines[i]); i += 1
            if ''.join(buf).strip():
                segs.append(('prose', ''.join(buf).strip(), start))
    return segs


def classify(title):
    """
    Classifies a section header as a Part, Item, or subsection marker.
    """

    p = _CONFIG.part_re.match(title)
    if p:
        return 'part', p.group(1).upper()
    it = _CONFIG.item_re.match(title)
    if it:
        return 'item', it.group(1).upper()
    return 'sub', title


def section_aware_chunks(text, size=256):
    """
    Splits text within Markdown sections while preserving section metadata.

    Tables are kept whole. Prose is split with the same sentence splitter used
    by sentence-aware chunking, but chunks do not cross section boundaries.
    """

    segs = segment(text)

    has_headers = any(kind == "header" for kind, content, start in segs)

    if not has_headers:
        chunks = sentence_aware_chunks(text, size=size)
        for c in chunks:
            c["strategy"] = "section"
        return chunks
    
    splitter = _sentence_splitter(size)
    raw = []
    part = item = header = None
    for kind, content, start in segment(text):
        if kind == 'header':
            m = _CONFIG.header_line.match(content)
            title = m.group(1).strip() if m else content.lstrip('#').strip()
            htype, val = classify(title)
            if htype == 'part':
                part, item, header = val, None, title
            elif htype == 'item':
                item, header = val, title
            else:
                header = title
            continue

        prefix = f'[{header}] ' if header else ''
        if kind == 'table':
            raw.append((prefix + content, part, item, header, True, start))
        else:
            for piece in splitter.split_text(content):
                raw.append((prefix + piece, part, item, header, False, start))

    return [make_chunk(i, t, "section", size, p, it, h, tbl, st)
            for i, (t, p, it, h, tbl, st) in enumerate(raw)]


def chunk_document(text, strategy="section", size=512):
    """
    Dispatches a document to the selected chunking strategy.
    """

    if strategy == "fixed":
        return fixed_size_chunks(text, size=size)

    if strategy == "sentence":
        return sentence_aware_chunks(text, size=size)

    if strategy == "section":
        return section_aware_chunks(text, size=size)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


