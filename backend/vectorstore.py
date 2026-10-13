import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"   # folder where Chroma saves its data
COLLECTION = "acme"         # like a table name inside the database


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks):
    """Embed all chunks and save them. Deletes any old index first."""
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)  # start fresh, avoid duplicates
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # measure similarity with cosine
    )


def load_vectorstore():
    """Open the index that build_vectorstore saved earlier."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )