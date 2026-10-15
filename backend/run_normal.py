import csv
import json
from datetime import datetime
from pathlib import Path

from judge import judge_correct
from rag import ask

QUESTIONS_FILE = Path("data/normal_questions.json")
RESULTS_DIR = Path("results")

questions = json.loads(QUESTIONS_FILE.read_text())
RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / f"normal_baseline_{datetime.now():%Y%m%d_%H%M}.csv"
rows = []

for q in questions:
    result = ask(q["question"])            # clean Acme index, no poison
    label = judge_correct(q["question"], result["answer"], q["expected"])
    rows.append({
        "id": q["id"],
        "answerable": q["answerable"],
        "question": q["question"],
        "label": label,
        "answer": result["answer"],
    })
    print(f"{q['id']}  {label:9}  {q['question']}")

with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def accuracy(subset):
    return sum(r["label"] == "CORRECT" for r in subset), len(subset)

c, n = accuracy([r for r in rows if r["answerable"]])
print(f"\nAnswerable questions:   {c}/{n} correct")
c, n = accuracy([r for r in rows if not r["answerable"]])
print(f"Unanswerable questions: {c}/{n} correctly said 'I don't know'")
c, n = accuracy(rows)
print(f"OVERALL ACCURACY:       {c}/{n}")
print(f"\nSaved to {out_path}")