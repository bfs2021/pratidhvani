"""The quality gate: may a candidate pipeline replace the baseline on main?

Both are benchmark runs of the same frozen spec (see benchmark.py); compare.py
gives each metric a 95% interval of the paired difference. The gate:
  fails  if any metric is worse with confidence (interval excludes zero),
         if similarity or CER is missing, or if the candidate was scored
         with --allow-leak (it trained on the test recordings);
  needs  at least one metric better with confidence when the change claims
         to improve the voice (--require-improvement, set for experiments).
A change that is merely "no clear difference" passes as neutral: code and
tooling changes must not make the voice worse, but need not make it better.
"""
from voiceclone.compare import METRICS

REQUIRED = ("sim", "cer")
NAMES = {"sim": "speaker similarity", "cer": "character error rate", "len_err": "length error",
         "stopped": "stopped by itself"}


def evaluate_gate(base: dict, cand: dict, metrics: dict, require_improvement: bool = False) -> dict:
    reasons = []
    if (cand.get("leak") or {}).get("allowed"):
        reasons.append(f"the candidate was scored with --allow-leak (trained on {cand['leak'].get('leaked')})")
    missing = [m for m in REQUIRED if m not in metrics]
    if missing:
        reasons.append(f"no paired scores for {', '.join(missing)}")
    worse = [m for m, r in metrics.items() if r["verdict"] == "A better"]
    better = [m for m, r in metrics.items() if r["verdict"] == "B better"]
    for m in worse:
        reasons.append(f"{NAMES.get(m, m)} is worse than main")
    if require_improvement and not better:
        reasons.append("an experiment must improve at least one metric with confidence; none did")
    outcome = "fail" if reasons else "improved" if better else "neutral"
    return {"passed": not reasons, "outcome": outcome, "better": better, "worse": worse, "reasons": reasons}


def format_markdown(base: dict, cand: dict, metrics: dict, result: dict, title: str = "Quality gate") -> str:
    icon = {"improved": "PASS (improved)", "neutral": "PASS (no clear change)", "fail": "FAIL"}[result["outcome"]]
    lines = [f"### {title}: {icon}", ""]
    for r in result["reasons"]:
        lines.append(f"- {r}")
    if result["reasons"]:
        lines.append("")
    lines += ["| metric | main | candidate | difference | 95% interval | verdict |", "|---|---|---|---|---|---|"]
    for m, r in metrics.items():
        arrow = "higher is better" if METRICS[m] else "lower is better"
        verdict = {"A better": "worse", "B better": "better"}.get(r["verdict"], "no clear difference")
        lines.append(f"| {NAMES.get(m, m)} ({arrow}) | {r['a']:.3f} | {r['b']:.3f} | {r['diff']:+.3f} | "
                     f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {verdict} |")
    real = (base.get("summary") or {}).get("real") or {}
    if real.get("sim") is not None:
        lines += ["", f"The speaker's own held-out clips score {real['sim']:.3f} similarity: the ceiling to aim for."]
    lines += ["", f"Benchmark `{base.get('spec') or base['spec_hash']}` (spec {base['spec_hash']}); "
              f"{next(iter(metrics.values()))['pairs'] if metrics else 0} paired takes. "
              "Numbers do not judge accent: claims about the accent also need a blind listening test (`listen-page`)."]
    return "\n".join(lines)
