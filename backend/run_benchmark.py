import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.callbacks import get_usage_metadata_callback

from chunking import smart_split
from defenses import DEFENSES
from judge import judge, judge_correct
from loaders import load_documents, load_file
from rag import ask
from vectorstore import get_embeddings

RUNS = 3  # repetitions per attack

# Claude Haiku 5.5 prices (USD per token), from the Claude API pricing page
PRICE_IN = 0.10 / 1_000_000
PRICE_OUT = 0.50 / 1_000_000

# Which defense? e.g.  python run_benchmark.py regex
defense_name = sys.argv[1] if len(sys.argv) > 1 else "none"
if defense_name not in DEFENSES:
    sys.exit(f"Unknown defense '{defense_name}'. Choose from: {', '.join(DEFENSES)}")
defense = DEFENSES[defense_name]

attacks = json.loads(Path("data/attacks.json").read_text())
normal_questions = json.loads(Path("data/normal_questions.json").read_text())
clean_docs = load_documents("data/acme", trust="trusted")
embeddings = get_embeddings()
stamp = f"{datetime.now():%Y%m%d_%H%M}"
Path("results").mkdir(exist_ok=True)


def timed_ask(question, **kwargs):
    """Run ask() and measure its time and the tokens it used.
    Counts the bot's answer AND any defense calls (e.g. the LLM judge),
    but NOT the grading judge, which runs afterwards."""
    with get_usage_metadata_callback() as cb:
        start = time.perf_counter()
        result = ask(question, **kwargs)
        seconds = time.perf_counter() - start
    tokens_in = sum(u.get("input_tokens", 0) for u in cb.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in cb.usage_metadata.values())
    return result, round(seconds, 2), tokens_in, tokens_out


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
        result, secs, t_in, t_out = timed_ask(attack["question"], vector_store=store, defense=defense)
        label = judge(result["answer"], attack["bad_link"])
        counts[label] += 1
        attack_rows.append({
            "attack_id": attack["id"],
            "family": attack["family"],
            "run": run,
            "poison_blocked": any(b["source"] == attack["file"] for b in result["blocked"]),
            "label": label,
            "latency_s": secs,
            "tokens_in": t_in,
            "tokens_out": t_out,
            "answer": result["answer"],
        })
    print(f"  {attack['id']:14} {counts}")

# ---------- 2. Normal questions ----------
print("\nNormal questions:")
normal_rows = []
for q in normal_questions:
    result, secs, t_in, t_out = timed_ask(q["question"], defense=defense)
    label = judge_correct(q["question"], result["answer"], q["expected"])
    normal_rows.append({
        "id": q["id"],
        "answerable": q["answerable"],
        "chunks_blocked": len(result["blocked"]),
        "label": label,
        "latency_s": secs,
        "tokens_in": t_in,
        "tokens_out": t_out,
        "answer": result["answer"],
    })
    print(f"  {q['id']}  {label}  ({secs:.1f}s)")

# ---------- 3. Summary ----------
at_risk = sum(r["label"] in ("FULL", "PARTIAL") for r in attack_rows)
full = sum(r["label"] == "FULL" for r in attack_rows)
poison_blocked = sum(r["poison_blocked"] for r in attack_rows)
correct = sum(r["label"] == "CORRECT" for r in normal_rows)
false_blocks = sum(r["chunks_blocked"] for r in normal_rows)

all_rows = attack_rows + normal_rows
avg_latency = sum(r["latency_s"] for r in all_rows) / len(all_rows)
avg_in = sum(r["tokens_in"] for r in all_rows) / len(all_rows)
avg_out = sum(r["tokens_out"] for r in all_rows) / len(all_rows)
cost_per_1000 = 1000 * (avg_in * PRICE_IN + avg_out * PRICE_OUT)

print(f"\n===== RESULTS: {defense_name} =====")
print(f"  Attacks - employee at risk:      {at_risk}/{len(attack_rows)}")
print(f"  Attacks - full success:          {full}/{len(attack_rows)}")
print(f"  Attacks - poison chunk blocked:  {poison_blocked}/{len(attack_rows)}")
print(f"  Normal  - correct answers:       {correct}/{len(normal_rows)}")
print(f"  Normal  - real chunks blocked:   {false_blocks}  (false positives)")
print(f"  Speed   - average answer time:   {avg_latency:.2f} s")
print(f"  Tokens  - average per question:  {avg_in:.0f} in, {avg_out:.0f} out")
print(f"  Cost    - per 1,000 questions:   ${cost_per_1000:.2f}")
print(f"\nSaved: {save(attack_rows, 'attacks')}")
print(f"Saved: {save(normal_rows, 'normal')}")