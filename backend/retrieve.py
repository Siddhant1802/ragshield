from vectorstore import load_vectorstore

# Open the index that build_index.py saved (no rebuilding needed)
store = load_vectorstore()

questions = [
    "I forgot my login, what should I do?",
    "How many vacation days can I carry over?",
    "Can I expense alcohol at a client dinner?",
    "Who won the football World Cup in 1998?",   # not in any document
]

# 1. Search WITH scores: see how relevant each chunk really is
for q in questions:
    print("Q:", q)
    results = store.similarity_search_with_relevance_scores(q, k=3)
    for doc, score in results:
        section = doc.metadata.get("h2", "-")
        print(f"   {score:.3f}  [{doc.metadata['source']} | {section}]")
    print()

# 2. The RETRIEVER: LangChain's standard "find relevant chunks" component.
#    Chains in Step 6 use this instead of calling the store directly.
retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How do I report a lost laptop?")
print("Retriever returned:", [d.metadata.get("h2") for d in docs])