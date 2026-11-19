"""Blind A/B listening page for two benchmark runs.

The page is one self-contained HTML file (audio embedded) that can be sent to
a listener. Each trial plays the two runs' takes of the same sentence and
seed as "1" and "2" in random order; every few trials a hidden check pairs a
real held-out recording with a clone take. The listener answers two
questions and downloads the answers as JSON. Which side was which run lives
only in the separate .key.json, which stays with you; `voiceclone compare
--key KEY --answers ANSWERS` turns the answers into win rates.
"""
import base64
import hashlib
import json
import random
from pathlib import Path

from voiceclone.compare import load_run
from voiceclone.jsonl import write_json


def build_trials(a: dict, b: dict, check_every: int = 6, seed: int = 0) -> list[dict]:
    """Trials in listening order: {id, kind, "1": label, "2": label, path1, path2, text, seed}.
    Labels are "A", "B" or "real"."""
    if a["spec_hash"] != b["spec_hash"]:
        raise ValueError("runs used different benchmark specs")
    rng = random.Random(seed)
    tb = {(t["text"], t["seed"]): t for t in b["takes"]}
    trials = []
    for t in a["takes"]:
        other = tb.get((t["text"], t["seed"]))
        if other is None:
            continue
        sides = [("A", t["path"]), ("B", other["path"])]
        rng.shuffle(sides)
        trials.append({"kind": "pair", "text": t["text"], "seed": t["seed"], "sides": sides})
    if not trials:
        raise ValueError("the two runs share no (sentence, seed) takes")
    n_check = len(trials) // check_every if check_every else 0
    clones = [("A", t["path"]) for t in a["takes"]] + [("B", t["path"]) for t in b["takes"]]
    for real in rng.sample(a["real"], min(n_check, len(a["real"]))):
        sides = [("real", real["path"]), rng.choice(clones)]
        rng.shuffle(sides)
        trials.append({"kind": "check", "text": real["text"], "seed": None, "sides": sides})
    rng.shuffle(trials)
    return [{"id": f"t{i:02d}", "kind": t["kind"], "text": t["text"], "seed": t["seed"],
             "1": t["sides"][0][0], "2": t["sides"][1][0], "path1": t["sides"][0][1], "path2": t["sides"][1][1]}
            for i, t in enumerate(trials)]


def _data_uri(path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def write_listening_page(run_a, run_b, out_html, check_every: int = 6, seed: int = 0, log=print):
    """Writes out_html and out_html's .key.json; returns both paths."""
    a, b = load_run(run_a), load_run(run_b)
    trials = build_trials(a, b, check_every, seed)
    page_id = hashlib.sha256(json.dumps([a["spec_hash"], str(run_a), str(run_b), seed]).encode()).hexdigest()[:12]
    out = Path(out_html); out.parent.mkdir(parents=True, exist_ok=True)
    page = [{"id": t["id"], "audio": [_data_uri(t["path1"]), _data_uri(t["path2"])]} for t in trials]
    out.write_text(PAGE.replace("__PAGE_ID__", page_id)
                   .replace("__TRIALS__", json.dumps(page).replace("</", "<\\/")), encoding="utf-8")
    key = out.with_suffix(".key.json")
    write_json(key, {"page_id": page_id, "run_a": str(run_a), "run_b": str(run_b), "spec_hash": a["spec_hash"],
                     "trials": {t["id"]: {k: v for k, v in t.items() if k != "id"} for t in trials}})
    n_check = sum(t["kind"] == "check" for t in trials)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(trials)} trials, {n_check} checks); "
        f"keep {key} to score the answers")
    return out, key


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test</title>
<style>
:root { --bg: #fbfaf7; --fg: #1d1c1a; --muted: #6b6760; --card: #ffffff; --line: #e3dfd6; --accent: #8a4b14; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #171614; --fg: #ece9e2; --muted: #a39e94; --card: #211f1c; --line: #38352f; --accent: #e0a060; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; }
.muted { color: var(--muted); }
.trial { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
.trial h2 { font-size: 1rem; margin: 0 0 12px; }
.players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
@media (max-width: 520px) { .players { grid-template-columns: 1fr; } }
.players div { display: flex; flex-direction: column; gap: 4px; }
audio { width: 100%; }
fieldset { border: 0; padding: 0; margin: 8px 0; }
legend { font-weight: 600; padding: 0; }
legend span { display: block; font-weight: 400; color: var(--muted); font-size: .9rem; }
label { margin-right: 16px; white-space: nowrap; }
.bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
       padding: 12px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: center; }
button { background: var(--accent); color: var(--bg); border: 0; border-radius: 6px; padding: 8px 16px; font: inherit; cursor: pointer; }
input[type=text] { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--card); color: var(--fg); }
</style></head>
<body><main>
<h1>Which recording sounds more like him?</h1>
<p class="muted">Each item has two recordings, 1 and 2. Listen to both, then answer both questions.
The order is random. If you really cannot tell, choose "can't tell". When done, press
<b>Download answers</b> and send the file back.</p>
<p><label>Your initials <input type="text" id="listener" maxlength="20"></label></p>
<div id="trials"></div>
</main>
<div class="bar"><span id="progress" class="muted"></span><button id="save">Download answers</button></div>
<script>
const PAGE_ID = "__PAGE_ID__";
const TRIALS = __TRIALS__;
const QUESTIONS = [
  ["voice", "Which sounds more like him?", "ಯಾವುದು ಅವರ ಧ್ವನಿಯಂತೆ ಹೆಚ್ಚು ಕೇಳಿಸುತ್ತದೆ?"],
  ["natural", "Which sounds like natural northern Kannada?", "ಯಾವುದು ಸಹಜ ಉತ್ತರ ಕರ್ನಾಟಕದ ಕನ್ನಡದಂತೆ ಕೇಳಿಸುತ್ತದೆ?"],
];
const STORE = "listen-" + PAGE_ID;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
const answers = saved.answers || {};
const root = document.getElementById("trials");
TRIALS.forEach((t, i) => {
  const box = document.createElement("section");
  box.className = "trial";
  let h = `<h2>Item ${i + 1} of ${TRIALS.length}</h2><div class="players">`;
  t.audio.forEach((src, j) => { h += `<div><b>${j + 1}</b><audio controls preload="none" src="${src}"></audio></div>`; });
  h += "</div>";
  QUESTIONS.forEach(([q, en, kn]) => {
    h += `<fieldset><legend>${en}<span>${kn}</span></legend>`;
    [["1", "1"], ["2", "2"], ["same", "can't tell"]].forEach(([v, lab]) => {
      const c = (answers[t.id] || {})[q] === v ? " checked" : "";
      h += `<label><input type="radio" name="${t.id}-${q}" value="${v}"${c}> ${lab}</label>`;
    });
    h += "</fieldset>";
  });
  box.innerHTML = h;
  box.addEventListener("change", e => {
    const [id, q] = e.target.name.split("-");
    (answers[id] = answers[id] || {})[q] = e.target.value;
    persist();
  });
  root.appendChild(box);
});
const who = document.getElementById("listener");
who.value = saved.listener || "";
who.addEventListener("input", persist);
function done() { return TRIALS.filter(t => answers[t.id] && answers[t.id].voice && answers[t.id].natural).length; }
function persist() {
  try { localStorage.setItem(STORE, JSON.stringify({ answers, listener: who.value })); } catch (e) {}
  document.getElementById("progress").textContent = `${done()} of ${TRIALS.length} answered`;
}
persist();
document.getElementById("save").addEventListener("click", () => {
  if (done() < TRIALS.length && !confirm(`Only ${done()} of ${TRIALS.length} answered. Download anyway?`)) return;
  const out = { page_id: PAGE_ID, listener: who.value.trim(), time: new Date().toISOString(),
    answers: TRIALS.filter(t => answers[t.id]).map(t => ({ trial: t.id, ...answers[t.id] })) };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = `answers_${PAGE_ID}_${out.listener || "anon"}.json`;
  a.click();
});
</script></body></html>
"""
