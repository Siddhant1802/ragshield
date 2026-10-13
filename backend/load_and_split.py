from chunking import smart_split
from loaders import load_documents

# 1. LOAD: read every supported file (.md, .txt, .pdf, .docx) in data/acme
docs = load_documents("data/acme")

# 2. SPLIT: let the system choose the best strategy and size for each document
chunks = smart_split(docs)

print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks\n")

# 3. LOOK: one line per chunk, showing what the system decided
for c in chunks:
    m = c.metadata
    section = m.get("h2", m.get("h1", "-"))
    print(f"[{m['source']}] section={section} | strategy={m['strategy']} "
          f"| size={m['chunk_size']} | {len(c.page_content)} chars")