import csv
import glob
import math
from pathlib import Path

ORDER = ["none", "input", "regex", "classifier", "isolation",
         "allowlist", "trust", "llm_judge", "full"]
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # Claude Haiku 5.5, USD per token


def latest(kind, defense):
    """Newest results file for this defense (e.g. results/attacks_regex_*.csv)."""
    files = sorted(glob.glob(f"results/{kind}_{defense}_*.csv"))
    return files[-1] if files else None


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    """95% confidence range for a rate k/n (Wilson score interval)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def avg(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return sum(vals) / len(vals) if vals else float("nan")


lines = ["| Defense | At risk | 95% range | Full success | Poison blocked | Normal correct | False positives | Avg time (s) | Cost / 1,000 |",
         "|---|---|---|---|---|---|---|---|---|"]
families = {}
done = []

for d in ORDER:
    a_path, n_path = latest("attacks", d), latest("normal", d)
    if not a_path or not n_path:
        print(f"(no results yet for '{d}')")
        continue
    done.append(d)
    attacks, normal = read(a_path), read(n_path)
    n = len(attacks)
    at_risk = sum(r["label"] in ("FULL", "PARTIAL") for r in attacks)
    full = sum(r["label"] == "FULL" for r in attacks)
    blocked = sum(r["poison_blocked"] == "True" for r in attacks)
    correct = sum(r["label"] == "CORRECT" for r in normal)
    fps = sum(int(r["chunks_blocked"]) for r in normal)
    lo, hi = wilson(at_risk, n)
    all_rows = attacks + normal
    cost = 1000 * (avg(all_rows, "tokens_in") * PRICE_IN + avg(all_rows, "tokens_out") * PRICE_OUT)
    lines.append(f"| {d} | {at_risk}/{n} ({at_risk/n:.0%}) | {lo:.0%}–{hi:.0%} | {full}/{n} | "
                 f"{blocked}/{n} | {correct}/{len(normal)} | {fps} | "
                 f"{avg(all_rows, 'latency_s'):.2f} | ${cost:.2f} |")
    for fam in sorted({r["family"] for r in attacks}):
        rows = [r for r in attacks if r["family"] == fam]
        k = sum(r["label"] in ("FULL", "PARTIAL") for r in rows)
        families.setdefault(fam, {})[d] = f"{k}/{len(rows)}"

lines += ["", "**At risk, by attack family**", "",
          "| Family | " + " | ".join(done) + " |",
          "|---|" + "---|" * len(done)]
for fam, by_def in families.items():
    lines.append(f"| {fam} | " + " | ".join(by_def.get(d, "-") for d in done) + " |")

table = "\n".join(lines)
print(table)
Path("results/summary.md").write_text(table + "\n")
print("\nSaved to results/summary.md")