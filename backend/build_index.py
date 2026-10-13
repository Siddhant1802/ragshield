from chunking import smart_split
from loaders import load_documents
from vectorstore import build_vectorstore

# Load -> split -> embed -> store
docs = load_documents("data/acme")
chunks = smart_split(docs)
store = build_vectorstore(chunks)
print(f"Stored {len(chunks)} chunks in Chroma\n")

# Quick test: a question that uses DIFFERENT words from the documents
question = "I forgot my login, what should I do?"
results = store.similarity_search(question, k=2)

print("Question:", question, "\n")
for r in results:
    section = r.metadata.get("h2", "-")
    print(f"[{r.metadata['source']} | {section}]")
    print(r.page_content[:150], "...\n")