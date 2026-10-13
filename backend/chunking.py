import re
import statistics

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Limits for chunk size, in characters.
# MAX: the embedding model reads ~256 tokens (~1000 characters); longer text gets cut off.
# MIN: smaller chunks lose too much context to be useful.
MIN_SIZE = 300
MAX_SIZE = 1000


def has_headings(text: str) -> bool:
    """True if the document has at least 2 markdown headings (#, ##, ###)."""
    return len(re.findall(r"^#{1,3} ", text, flags=re.MULTILINE)) >= 2


def pick_chunk_size(text: str) -> int:
    """Choose a chunk size of about 3 typical paragraphs, kept within MIN..MAX."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return 500
    median = statistics.median(len(p) for p in paragraphs)
    return int(min(MAX_SIZE, max(MIN_SIZE, median * 3)))


def smart_split(docs: list[Document]) -> list[Document]:
    """Split each document using the best strategy for that document."""
    chunks = []
    for doc in docs:
        text = doc.page_content
        size = pick_chunk_size(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10)

        if has_headings(text):
            # Rule 1: split by headings, one section per piece
            strategy = "headings"
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
                strip_headers=False,  # keep the heading text inside the chunk
            )
            sections = header_splitter.split_text(text)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}  # keep source + add headings
        else:
            # Rule 2: no headings, so treat the whole document as one piece
            strategy = "paragraphs"
            sections = [doc]

        # Rule 3: any piece still too long gets split again at the chosen size
        for c in splitter.split_documents(sections):
            c.metadata["strategy"] = strategy
            c.metadata["chunk_size"] = size
            chunks.append(c)
    return chunks