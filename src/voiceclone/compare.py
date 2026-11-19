"""Is benchmark run B better than run A?

Takes are paired by sentence and seed, so each difference compares the same
words under the same sampling seed. The 95% interval of the mean difference
comes from a bootstrap that resamples whole sentences (with all their seeds):
takes of one sentence share its difficulty, so resampling them one by one
would make the interval too narrow. A metric only gets "B better" or
"A better" when its interval excludes zero.
"""
import json
from pathlib import Path

import numpy as np

# metric -> True if higher is better
METRICS = {"sim": True, "cer": False, "len_err": False, "stopped": True}


def load_run(run_dir) -> dict:
    p = Path(run_dir) / "scores.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: is {run_dir} a `voiceclone benchmark` output folder?")
    return json.load(open(p, encoding="utf-8"))


def paired_differences(a: dict, b: dict, metric: str) -> dict[str, list[float]]:
    """sentence -> [B - A for each seed], pairing takes by (sentence, seed)."""
    if a["spec_hash"] != b["spec_hash"]:
        raise ValueError(f"runs used different benchmark specs ({a['spec_hash']} vs {b['spec_hash']}); "
                         "their scores are not comparable")
    ta = {(t["text"], t["seed"]): t for t in a["takes"]}
    tb = {(t["text"], t["seed"]): t for t in b["takes"]}
    if ta.keys() != tb.keys():
        raise ValueError(f"runs have different takes: {len(ta.keys() - tb.keys())} only in A, "
                         f"{len(tb.keys() - ta.keys())} only in B (an interrupted benchmark?)")
    out: dict[str, list[float]] = {}
    for k in sorted(ta, key=lambda k: (k[0], k[1])):
        va, vb = ta[k].get(metric), tb[k].get(metric)
        if va is not None and vb is not None:
            out.setdefault(k[0], []).append(float(vb) - float(va))
    return out


def bootstrap_ci(groups: list[list[float]], n: int = 10000, seed: int = 0, level: float = 0.95):
    """(mean, low, high) of the mean over all values, resampling whole groups."""
    sums = np.array([sum(g) for g in groups]); counts = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(n, len(groups)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(sums.sum() / counts.sum()), float(lo), float(hi)


def verdict(lo: float, hi: float, higher_is_better: bool) -> str:
    """For an interval of B - A."""
    if lo > 0:
        return "B better" if higher_is_better else "A better"
    if hi < 0:
        return "A better" if higher_is_better else "B better"
    return "no clear difference"


def compare_runs(a: dict, b: dict, n_boot: int = 10000) -> dict:
    res = {}
    for m, higher in METRICS.items():
        groups = list(paired_differences(a, b, m).values())
        if not groups:
            continue
        mean, lo, hi = bootstrap_ci(groups, n_boot)
        res[m] = {"a": _mean(a["takes"], m), "b": _mean(b["takes"], m), "diff": mean, "ci": [lo, hi],
                  "pairs": sum(map(len, groups)), "sentences": len(groups), "verdict": verdict(lo, hi, higher)}
    return res


def _mean(takes, k):
    v = [float(t[k]) for t in takes if t.get(k) is not None]
    return float(np.mean(v)) if v else None


# ---------- blind listening answers ----------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval of a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def listening_results(key: dict, answer_files: list[dict], run_a, run_b) -> dict:
    """Win rates of B over A per question, from the answers of one or more listeners.

    `key` is the .key.json the page generator wrote; it says which side of
    each trial was which run. RUN_A/RUN_B may be given in either order.
    "Can't tell" answers count as neither. Check trials (a real clip against
    a clone) measure how reliable each listener is: a listener who cannot
    pick the real recording is guessing.
    """
    same = lambda x, y: Path(x).resolve() == Path(y).resolve()
    if same(run_a, key["run_a"]) and same(run_b, key["run_b"]):
        label = {"A": "A", "B": "B", "real": "real"}
    elif same(run_a, key["run_b"]) and same(run_b, key["run_a"]):
        label = {"A": "B", "B": "A", "real": "real"}
    else:
        raise ValueError(f"the listening key is for {key['run_a']} vs {key['run_b']}, not these runs")
    out = {"listeners": [], "questions": {}}
    tally = {q: {"A": 0, "B": 0, "tie": 0} for q in ("voice", "natural")}
    for ans in answer_files:
        if ans.get("page_id") != key["page_id"]:
            raise ValueError(f"answers for page {ans.get('page_id')}, key is for page {key['page_id']}")
        check = {q: [0, 0] for q in ("voice", "natural")}
        for a in ans["answers"]:
            t = key["trials"][a["trial"]]
            for q in ("voice", "natural"):
                pick = a.get(q)
                if pick not in ("1", "2"):
                    if t["kind"] == "pair" and pick:
                        tally[q]["tie"] += 1
                    continue
                won = label[t[pick]]
                if t["kind"] == "check":
                    check[q][0] += won == "real"; check[q][1] += 1
                else:
                    tally[q][won] += 1
        out["listeners"].append({"listener": ans.get("listener") or "?", "answered": len(ans["answers"]),
                                 "picked_real": {q: f"{k}/{n}" for q, (k, n) in check.items()}})
    for q, t in tally.items():
        n = t["A"] + t["B"]
        lo, hi = wilson(t["B"], n)
        out["questions"][q] = {**t, "b_win_rate": t["B"] / n if n else None, "ci": [lo, hi],
                               "verdict": "B preferred" if lo > 0.5 else "A preferred" if hi < 0.5
                               else "no clear preference"}
    return out


QUESTION_TEXT = {"voice": "sounds more like him", "natural": "natural northern Kannada"}


def format_report(a: dict, b: dict, metrics: dict, listening: dict | None = None) -> str:
    lines = [f"A = {a['run']} ({a.get('adapter') or 'zero-shot'})", f"B = {b['run']} ({b.get('adapter') or 'zero-shot'})"]
    for r in (a, b):
        if (r.get("leak") or {}).get("allowed"):
            lines.append(f"WARNING: {r['run']} was scored with --allow-leak (trained on {r['leak']['leaked']})")
    lines.append(f"{'metric':8s} {'A':>7s} {'B':>7s} {'B-A':>8s}  95% interval        pairs  verdict")
    for m, r in metrics.items():
        lines.append(f"{m:8s} {r['a']:7.3f} {r['b']:7.3f} {r['diff']:+8.3f}  [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
                     f"  {r['pairs']:5d}  {r['verdict']}")
    if listening:
        for q, r in listening["questions"].items():
            rate = "?" if r["b_win_rate"] is None else f"{r['b_win_rate']:.0%}"
            lines.append(f"listening, {QUESTION_TEXT[q]}: B won {r['B']} of {r['A'] + r['B']} ({rate}, "
                         f"95% {r['ci'][0]:.0%}-{r['ci'][1]:.0%}), {r['tie']} can't tell -> {r['verdict']}")
        for li in listening["listeners"]:
            lines.append(f"  listener {li['listener']}: picked the real recording "
                         f"{li['picked_real']['voice']} (voice), {li['picked_real']['natural']} (natural)")
    return "\n".join(lines)
