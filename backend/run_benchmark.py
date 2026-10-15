import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from chunking import smart_split
from defenses import DEFENSES
from judge import judge, judge_correct
from loaders import load_documents, load_file
from rag import ask
from vectorstore import get_embeddings

RUNS = 5  # repetitions per attack

# Which defense? e.g.  python run_benchmark.py regex
defense_name = sys.argv[1] if len(sys.argv) > 1 else "none"
if defense_name not in DEFENSES:
    sys.exit(f"Unknown defense '{defense_name}'. Choose from: {', '.join(DEFENSES)}")
defense = DEFENSES[defense_name]

attacks = json.loads(Path("data/attacks.json").read_text())
normal_questions = json.loads(Path("data/normal_questions.json").read_text())
clean_docs = load_documents("data/acme")
embeddings = get_embeddings()
stamp = f"{datetime.now():%Y%m%d_%H%M}"
Path("results").mkdir(exist_ok=True)


def save(rows, kind):
    path = Path("results") / f"{kind}_{defense_name}_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


print(f"===== Defense: {defense_name} =====\n")

# ---------- 1. Attacks ----------
print("Attacks:")
attack_rows = []
for attack in attacks:
    poison = load_file(Path("data/attacks") / attack["file"])
    store = Chroma.from_documents(
        smart_split(clean_docs + poison),
        embeddings,
        collection_name=f"bench_{defense_name}_{attack['id']}",
        collection_metadata={"hnsw:space": "cosine"},
    )
    counts = {"FULL": 0, "PARTIAL": 0, "RESISTED": 0, "UNKNOWN": 0}
    for run in range(1, RUNS + 1):
        result = ask(attack["question"], vector_store=store, defense=defense)
        label = judge(result["answer"], attack["bad_link"])
        counts[label] += 1
        attack_rows.append({
            "attack_id": attack["id"],
            "family": attack["family"],
            "run": run,
            "poison_blocked": any(b["source"] == attack["file"] for b in result["blocked"]),
            "label": label,
            "answer": result["answer"],
        })
    print(f"  {attack['id']:14} {counts}")

# ---------- 2. Normal questions ----------
print("\nNormal questions:")
normal_rows = []
for q in normal_questions:
    result = ask(q["question"], defense=defense)
    label = judge_correct(q["question"], result["answer"], q["expected"])
    normal_rows.append({
        "id": q["id"],
        "answerable": q["answerable"],
        "chunks_blocked": len(result["blocked"]),
        "label": label,
        "answer": result["answer"],
    })
    print(f"  {q['id']}  {label}")

# ---------- 3. Summary ----------
at_risk = sum(r["label"] in ("FULL", "PARTIAL") for r in attack_rows)
full = sum(r["label"] == "FULL" for r in attack_rows)
poison_blocked = sum(r["poison_blocked"] for r in attack_rows)
correct = sum(r["label"] == "CORRECT" for r in normal_rows)
false_blocks = sum(r["chunks_blocked"] for r in normal_rows)

print(f"\n===== RESULTS: {defense_name} =====")
print(f"  Attacks - employee at risk:      {at_risk}/{len(attack_rows)}")
print(f"  Attacks - full success:          {full}/{len(attack_rows)}")
print(f"  Attacks - poison chunk blocked:  {poison_blocked}/{len(attack_rows)}")
print(f"  Normal  - correct answers:       {correct}/{len(normal_rows)}")
print(f"  Normal  - real chunks blocked:   {false_blocks}  (false positives)")
print(f"\nSaved: {save(attack_rows, 'attacks')}")
print(f"Saved: {save(normal_rows, 'normal')}")