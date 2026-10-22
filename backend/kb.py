import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma

from chunking import smart_split
from loaders import SUPPORTED, load_documents, load_file
from vectorstore import get_embeddings

UPLOAD_DIR = Path("data/uploads")      # files uploaded through the app
TRUST_FILE = UPLOAD_DIR / "_trust.json"  # remembers each upload's trust level

_embeddings = get_embeddings()
_store: Chroma | None = None


def _trust_map() -> dict:
    if TRUST_FILE.exists():
        return json.loads(TRUST_FILE.read_text())
    return {}


def _save_trust(m: dict):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRUST_FILE.write_text(json.dumps(m, indent=2))


def _all_docs():
    """Clean Acme docs (trusted) + every uploaded file with its saved trust level."""
    docs = load_documents("data/acme", trust="trusted")
    trust = _trust_map()
    for path in sorted(UPLOAD_DIR.glob("*")):
        if path.name == "_trust.json" or path.suffix.lower() not in SUPPORTED:
            continue
        docs.extend(load_file(path, trust=trust.get(path.name, "untrusted")))
    return [d for d in docs if d.page_content.strip()]


def rebuild():
    """Rebuild the in-memory index from the current documents. Called at startup and after any change."""
    global _store
    chunks = smart_split(_all_docs())
    _store = Chroma.from_documents(
        chunks, _embeddings,
        collection_name="live_kb",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _store


def get_store() -> Chroma:
    return _store if _store is not None else rebuild()


def list_documents() -> list[dict]:
    """One row per document, for the Documents screen."""
    chunks = get_store().get()                       # all chunks in the index
    counts, trust = {}, {}
    for meta in chunks["metadatas"]:
        src = meta.get("source", "?")
        counts[src] = counts.get(src, 0) + 1
        trust[src] = meta.get("trust", "untrusted")
    acme = {p.name for p in Path("data/acme").glob("*")}
    rows = []
    for src in sorted(counts):
        rows.append({
            "name": src,
            "trust": trust[src],
            "chunks": counts[src],
            "origin": "built-in" if src in acme else "uploaded",
        })
    return rows


def add_document(filename: str, data: bytes, trust: str) -> dict:
    """Save an uploaded file, record its trust level, and rebuild the index."""
    if Path(filename).suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED))}")
    trust = "trusted" if trust == "trusted" else "untrusted"   # untrusted by default
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name                                  # strip any folder path
    (UPLOAD_DIR / safe).write_bytes(data)
    m = _trust_map(); m[safe] = trust; _save_trust(m)
    rebuild()
    return {"name": safe, "trust": trust}


def remove_document(filename: str) -> bool:
    """Delete an uploaded file (built-in Acme docs can't be removed)."""
    safe = Path(filename).name
    path = UPLOAD_DIR / safe
    if not path.exists():
        return False
    path.unlink()
    m = _trust_map(); m.pop(safe, None); _save_trust(m)
    rebuild()
    return True