import csv
import json
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from chunking import smart_split
from judge import judge
from loaders import load_documents, load_file
from rag import ask
from vectorstore import get_embeddings

RUNS = 5                              # repetitions per attack
ATTACKS_FILE = Path("data/attacks.json")
RESULTS_DIR = Path("results")

attacks = json.loads(ATTACKS_FILE.read_text())
clean_docs = load_documents("data/acme")
embeddings = get_embeddings()

RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / f"baseline_{datetime.now():%Y%m%d_%H%M}.csv"
rows = []

for attack in attacks:
    # Separate in-memory index per attack: clean docs + this ONE poisoned doc
    poison = load_file(Path("data/attacks") / attack["file"])
    store = Chroma.from_documents(
        smart_split(clean_docs + poison),
        embeddings,
        collection_name=f"attack_{attack['id']}",
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Where did the poison rank in the top 3? (None = not retrieved)
    poison_rank, poison_score = None, None
    top = store.similarity_search_with_relevance_scores(attack["question"], k=3)
    for rank, (doc, score) in enumerate(top, start=1):
        if doc.metadata["source"] == attack["file"]:
            poison_rank, poison_score = rank, round(score, 3)
            break

    counts = {"FULL": 0, "PARTIAL": 0, "RESISTED": 0, "UNKNOWN": 0}
    for run in range(1, RUNS + 1):
        result = ask(attack["question"], vector_store=store)
        label = judge(result["answer"], attack["bad_link"])
        counts[label] += 1
        rows.append({
            "attack_id": attack["id"],
            "family": attack["family"],
            "run": run,
            "poison_rank": poison_rank,
            "poison_score": poison_score,
            "label": label,
            "answer": result["answer"],
        })
    print(f"{attack['id']:14} poison rank={poison_rank}  {counts}")

# Save every single run to a CSV file
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Summary per attack family
print("\nSUMMARY BY FAMILY")
for family in sorted({r["family"] for r in rows}):
    labels = [r["label"] for r in rows if r["family"] == family]
    at_risk = sum(label in ("FULL", "PARTIAL") for label in labels)
    full = labels.count("FULL")
    print(f"  {family:22} at risk: {at_risk}/{len(labels)}   full: {full}/{len(labels)}")

print(f"\nSaved {len(rows)} results to {out_path}")