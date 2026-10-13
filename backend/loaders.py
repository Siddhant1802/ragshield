from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def load_file(path: Path) -> list[Document]:
    """Read one file into Document objects, based on its file type."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": path.name})]

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        # One Document per page, so answers can cite the page number
        return [
            Document(page_content=page.extract_text() or "",
                     metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]

    if suffix == ".docx":
        import docx
        d = docx.Document(path)
        text = "\n".join(p.text for p in d.paragraphs)
        return [Document(page_content=text, metadata={"source": path.name})]

    return []  # unsupported file type: skip it


def load_documents(folder: str) -> list[Document]:
    """Load every supported file in a folder (including subfolders)."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in SUPPORTED:
            docs.extend(load_file(path))
    # Drop empty documents, e.g. scanned PDF pages with no extractable text
    return [d for d in docs if d.page_content.strip()]