import csv
import glob
import math

ORDER = ["none", "input", "regex", "classifier", "isolation",
         "allowlist", "trust", "llm_judge", "full"]
LABELS = {
    "none": "No defense", "input": "Input guard", "regex": "Regex filter",
    "classifier": "Classifier", "isolation": "Prompt isolation",
    "allowlist": "Link allowlist", "trust": "Source trust",
    "llm_judge": "LLM judge", "full": "RAGShield full",
}
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # Claude Haiku 5.5


def _latest(kind, defense):
    files = sorted(glob.glob(f"results/{kind}_{defense}_*.csv"))
    return files[-1] if files else None


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 3), round(min(1, c + h), 3)


def _avg(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def summary() -> list[dict]:
    """One row per defense that has results, ready for the Attack Lab table/charts."""
    out = []
    for d in ORDER:
        a_path, n_path = _latest("attacks", d), _latest("normal", d)
        if not a_path or not n_path:
            continue
        attacks, normal = _read(a_path), _read(n_path)
        n = len(attacks)
        at_risk = sum(r["label"] in ("FULL", "PARTIAL") for r in attacks)
        full = sum(r["label"] == "FULL" for r in attacks)
        blocked = sum(r.get("poison_blocked") == "True" for r in attacks)
        correct = sum(r["label"] == "CORRECT" for r in normal)
        fps = sum(int(r["chunks_blocked"]) for r in normal)
        lo, hi = _wilson(at_risk, n)
        allr = attacks + normal
        cost = 1000 * (_avg(allr, "tokens_in") * PRICE_IN + _avg(allr, "tokens_out") * PRICE_OUT)
        out.append({
            "defense": d,
            "label": LABELS.get(d, d),
            "attack_runs": n,
            "at_risk": at_risk,
            "at_risk_pct": round(100 * at_risk / n, 1) if n else 0,
            "at_risk_ci": [round(100 * lo, 1), round(100 * hi, 1)],
            "full_success": full,
            "poison_blocked": blocked,
            "normal_total": len(normal),
            "normal_correct": correct,
            "false_positives": fps,
            "avg_latency_s": round(_avg(allr, "latency_s"), 2),
            "cost_per_1000": round(cost, 3),
        })
    return out