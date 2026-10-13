import math

from langchain_huggingface import HuggingFaceEmbeddings

# A small, free embedding model that runs on your Mac (downloads ~90 MB the first time)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "Steps to change my login credentials",
    "What is the meal allowance on business trips?",
]
vectors = emb.embed_documents(sentences)

print("Each sentence becomes", len(vectors[0]), "numbers")
print("First 5 numbers of sentence 1:", [round(x, 3) for x in vectors[0][:5]])


def cosine(a, b):
    """Similarity between two vectors: 1 = same meaning, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


print("\nSimilarity 1 vs 2 (password vs login):", round(cosine(vectors[0], vectors[1]), 3))
print("Similarity 1 vs 3 (password vs meals): ", round(cosine(vectors[0], vectors[2]), 3))