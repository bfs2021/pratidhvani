"""A frozen benchmark, so one cloning run can be compared with another.

The spec (JSON) fixes everything a score depends on except the adapter:
  holdout       a clips folder (metadata.jsonl) and the recordings (sources)
                whose clips form the speaker's voiceprint. Whole recordings:
                clips of a recording the model trained on score higher than
                clips of a new session, so they would flatter the clone.
  conditioning  the reference clip and its text, the same for every run
  sentences     what to say, inline so that editing a text file cannot
                change an existing benchmark
  seeds         each sentence is spoken once per seed (default 0, 1, 2)
  gen           sampling parameters (GenParams fields)
Its hash travels with every result; `compare` refuses runs of different specs.

Each take is scored against the held-out voiceprint (ECAPA cosine), by the
MMS recogniser (CER), by length (length ratio at natural_sps syllables/s) and
by whether generation stopped by itself or ran into the length cap. The real
held-out clips are scored the same way, their similarity against a centroid
of the other held-out clips, so a run is also reported as a fraction of what
the speaker himself achieves.

A recording's name is a clip's `source` field, or else its file name without
the chunk number (`prepare` names chunks TAG_NNNN.wav, so K06_0012 -> K06).
"""
import gc
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voiceclone.audio import TTS_SR, load16, write_wav
from voiceclone.jsonl import read_jsonl, write_json
from voiceclone.text import cer, romanize, syllables

DEFAULT_SEEDS = [0, 1, 2]
NATURAL_SPS = 3.2
MIN_TAKE_SEC = 0.3   # shorter takes are scored as empty


class LeakError(Exception):
    """The adapter trained on audio the benchmark scores against (or this cannot be ruled out)."""


def source_of(clip) -> str:
    """Recording a clip (metadata row or file name) came from."""
    if isinstance(clip, dict):
        if clip.get("source"):
            return str(clip["source"])
        clip = clip["file"]
    return re.sub(r"_\d+$", "", Path(clip).stem)


# ---------- spec ----------
def load_spec(path) -> dict:
    spec = json.load(open(path, encoding="utf-8"))
    missing = {"holdout", "conditioning", "sentences"} - spec.keys()
    if missing:
        raise ValueError(f"{path}: spec needs {sorted(missing)}")
    if not spec["holdout"].get("sources"):
        raise ValueError(f"{path}: holdout.sources must name at least one recording")
    spec.setdefault("seeds", DEFAULT_SEEDS)
    spec.setdefault("gen", {})
    spec.setdefault("natural_sps", NATURAL_SPS)
    spec["conditioning"].setdefault("source", source_of(spec["conditioning"]["wav"]))
    return spec


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def holdout_clips(spec: dict) -> list[dict]:
    """Held-out clips (with `path`), minus the conditioning clip itself."""
    h = spec["holdout"]
    d, want = Path(h["clips"]), set(h["sources"])
    cond = Path(spec["conditioning"]["wav"]).resolve()
    rows = [{**c, "path": str(d / c["file"])} for c in read_jsonl(d / "metadata.jsonl")
            if source_of(c) in want and (d / c["file"]).resolve() != cond]
    absent = want - {source_of(c) for c in rows}
    if absent:
        raise ValueError(f"no clips of {sorted(absent)} in {d}/metadata.jsonl")
    if h.get("max_clips"):
        random.Random(0).shuffle(rows)
        rows = rows[: h["max_clips"]]
    if len(rows) < 2:
        raise ValueError("need at least 2 held-out clips (each real clip is scored against the others)")
    return rows


# ---------- leakage guard ----------
def find_train_log(adapter) -> Path | None:
    """train_log.json of an adapter; a checkpoint-NNNNN/ keeps it in its parent."""
    for d in (Path(adapter), Path(adapter).parent):
        if (d / "train_log.json").exists():
            return d / "train_log.json"
    return None


def read_train_log(adapter) -> dict:
    p = find_train_log(adapter)
    tl = json.load(open(p, encoding="utf-8")) if p else {}
    tl.pop("losses", None)
    # older adapters wrote their arguments under "args"
    params = tl.get("params") or tl.get("args") or {}
    return {**tl, "params": params, "clips": tl.get("clips") or params.get("clips")}


def training_sources(adapter, train_clips=None) -> dict[str, int]:
    """Recording -> number of clips the adapter learned from, including every
    adapter it continued training (--init-adapter). `train_clips` (clip
    folders) replaces the lookup through train_log.json."""
    folders = [Path(d) for d in train_clips or []]
    todo = [] if folders else [str(adapter)]
    while todo:
        a = todo.pop()
        if find_train_log(a) is None:
            raise LeakError(f"{a}: no train_log.json in it or its parent, so its training data is unknown; "
                            "pass --train-clips")
        tl = read_train_log(a)
        if not tl["clips"]:
            raise LeakError(f"{find_train_log(a)} does not say which clips it trained on; pass --train-clips")
        folders.append(Path(tl["clips"]))
        if tl["params"].get("init_adapter"):
            todo.append(tl["params"]["init_adapter"])
    counts = Counter()
    for d in dict.fromkeys(f.resolve() for f in folders):
        if not (d / "metadata.jsonl").exists():
            raise LeakError(f"training clips {d} not found (paths in train_log.json are relative to where "
                            "train ran); pass --train-clips")
        counts.update(source_of(c) for c in read_jsonl(d / "metadata.jsonl"))
    return dict(counts)


def leaked_sources(spec: dict, trained: dict) -> list[str]:
    """Benchmark recordings (held-out or conditioning) that are among the trained ones."""
    used = set(spec["holdout"]["sources"]) | {spec["conditioning"]["source"]}
    return sorted(used & set(trained))


def leak_guard(spec: dict, adapter, train_clips=None, allow: bool = False, log=print) -> dict:
    """Raise LeakError unless the adapter provably never trained on benchmark audio
    (or `allow`). A zero-shot run (no adapter) cannot leak."""
    if not adapter and not train_clips:
        return {"trained_sources": {}, "leaked": [], "allowed": False}
    try:
        trained = training_sources(adapter, train_clips)
    except LeakError as e:
        if not allow:
            raise
        log(f"WARNING: leak check skipped: {e}")
        return {"trained_sources": None, "leaked": None, "allowed": True}
    leaked = leaked_sources(spec, trained)
    if leaked and not allow:
        raise LeakError(f"the adapter trained on {leaked}, which the benchmark scores against; "
                        "its scores would not be comparable (pass --allow-leak to score anyway)")
    if leaked:
        log(f"WARNING: scoring a leaked adapter (trained on {leaked}); results are flagged")
    return {"trained_sources": trained, "leaked": leaked, "allowed": bool(leaked)}


# ---------- synthesis ----------
def take_key(t: dict) -> tuple[str, int]:
    """What a take is compared on across runs: the sentence it says and its seed."""
    return t["text"], t["seed"]


def synthesize_takes(synth, spec: dict, out_dir, log=print) -> list[dict]:
    """seed_S/sample_XX.wav + log.json per seed. Resumes: a take already in log.json
    with the same text is kept, so an interrupted run continues where it stopped."""
    from voiceclone.svara.synth import GenParams
    from voiceclone.svara.tokens import token_budget

    p, cond = GenParams(**spec["gen"]), spec["conditioning"]
    ref_tokens, takes = None, []
    for seed in spec["seeds"]:
        d = Path(out_dir) / f"seed_{seed}"; d.mkdir(parents=True, exist_ok=True)
        done = {r["file"]: r for r in json.load(open(d / "log.json", encoding="utf-8"))} \
            if (d / "log.json").exists() else {}
        for i, text in enumerate(spec["sentences"]):
            name = f"sample_{i:02d}.wav"
            if not (done.get(name, {}).get("text") == text and (d / name).exists()):
                if ref_tokens is None:
                    ref_tokens = synth.reference(cond["wav"])
                t0 = time.time()
                wav, n = synth.generate(text, ref_tokens, cond["text"], seed=seed, p=p)
                write_wav(d / name, wav, TTS_SR)
                cap = token_budget(syllables(text), p.max_sec, p.min_sps, p.max_sps)[1]
                done[name] = {"file": name, "text": text, "seconds": round(len(wav) / TTS_SR, 2),
                              "gen_time": round(time.time() - t0, 1), "tokens": n, "stopped": n < cap}
                write_json(d / "log.json", sorted(done.values(), key=lambda r: r["file"]))
                log(f"seed {seed} [{i}] {done[name]['seconds']:.1f}s in {done[name]['gen_time']:.0f}s"
                    f"{'' if done[name]['stopped'] else ' (hit the length cap)'} | {text[:50]}")
            takes.append({**done[name], "sentence": i, "seed": seed, "path": str(d / name)})
    return takes


# ---------- scoring ----------
def length_ratio(seconds: float, text: str, natural_sps: float) -> float | None:
    expected = syllables(text) / natural_sps
    return float(seconds / expected) if expected else None


def score_real(encoder, aligner, clips: list[dict], natural_sps: float = NATURAL_SPS):
    """(voiceprint centroid, scored real clips). Each real clip's similarity is
    to the centroid of the other clips, so a clip never scores against itself."""
    embs = np.stack([encoder.embed(load16(c["path"])) for c in clips])
    total = embs.sum(0)
    rows = []
    for c, e in zip(clips, embs):
        loo = total - e
        y = load16(c["path"])
        rows.append({"file": c["file"], "source": source_of(c), "path": c["path"], "text": c["text"],
                     "sim": float(np.dot(e, loo / np.linalg.norm(loo))),
                     "cer": float(cer(aligner.greedy_decode(y), romanize(c["text"]))),
                     "len_ratio": length_ratio(len(y) / 16000, c["text"], natural_sps)})
    return total / np.linalg.norm(total), rows


def score_takes(encoder, aligner, centroid, takes: list[dict], real_len_ratio: float | None,
                natural_sps: float = NATURAL_SPS, log=print) -> list[dict]:
    """Adds sim, cer, len_ratio and len_err (|log| of length vs the real speaker's) to each take.
    A take with (almost) no audio scores as no voice and nothing heard, so a
    run cannot look better by failing on its hardest sentences."""
    out = []
    for t in takes:
        y = load16(t["path"])
        lr = length_ratio(len(y) / 16000, t["text"], natural_sps)
        empty = len(y) < MIN_TAKE_SEC * 16000
        s = {**t, "empty": empty, "len_ratio": lr,
             "sim": 0.0 if empty else float(np.dot(encoder.embed(y), centroid)),
             "cer": 1.0 if empty else float(cer(aligner.greedy_decode(y), romanize(t["text"]))),
             "len_err": abs(float(np.log(max(lr, 0.1) / real_len_ratio))) if lr is not None and real_len_ratio
             else None}
        log(f"seed {t['seed']} [{t['sentence']}] sim {s['sim']:.3f} cer {s['cer']:.2f} "
            f"len {'?' if lr is None else format(lr, '.2f')}")
        out.append(s)
    return out


def _mean(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def summarize(takes: list[dict], real: list[dict]) -> dict:
    """Raw means, the real clips' means, and the run relative to real
    (sim: higher is better, 1.0 = as close as his own recordings; cer and
    len_ratio: 1.0 = like his own recordings, cer above 1 is less clear)."""
    raw = {k: _mean(takes, k) for k in ("sim", "cer", "len_ratio", "len_err", "stopped")}
    ref = {k: _mean(real, k) for k in ("sim", "cer", "len_ratio")}
    rel = {k: raw[k] / ref[k] if raw[k] is not None and ref[k] else None for k in ("sim", "cer", "len_ratio")}
    return {"raw": raw, "real": ref, "relative": rel, "n_takes": len(takes), "n_real": len(real)}


def _free_memory():
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def run_benchmark(spec_path, out_dir, lora=None, model=None, allow_leak: bool = False, train_clips=None,
                  leaderboard=None, ecapa_dir="models/ecapa", log=print) -> dict:
    """Synthesize every sentence x seed, score, write scores.json and append a leaderboard row."""
    from voiceclone.svara.synth import DEFAULT_MODEL

    spec = load_spec(spec_path)
    h, model = spec_hash(spec), model or DEFAULT_MODEL
    leak = leak_guard(spec, lora, train_clips, allow_leak, log)   # before the slow part
    clips = holdout_clips(spec)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "spec.json", {**spec, "hash": h})

    from voiceclone.svara.synth import Synthesizer

    synth = Synthesizer(model, lora)
    takes = synthesize_takes(synth, spec, out, log)
    del synth; _free_memory()   # the TTS model and the scorers do not fit 16 GB together comfortably

    from voiceclone.align import Aligner
    from voiceclone.speaker import SpeakerEncoder

    enc, aligner = SpeakerEncoder(ecapa_dir), Aligner()
    cent, real = score_real(enc, aligner, clips, spec["natural_sps"])
    takes = score_takes(enc, aligner, cent, takes, _mean(real, "len_ratio"), spec["natural_sps"], log)
    summary = summarize(takes, real)
    tl = read_train_log(lora) if lora else {}
    result = {"run": out.name, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "spec": spec.get("name"), "spec_hash": h, "adapter": lora, "model": model,
              "train": {k: tl.get(k) for k in ("params", "steps", "clips", "model")} if tl else None,
              "leak": leak, "summary": summary, "takes": takes, "real": real}
    write_json(out / "scores.json", result)
    lb = Path(leaderboard) if leaderboard else out.parent / "results.jsonl"
    with open(lb, "a", encoding="utf-8") as f:
        f.write(json.dumps({**{k: v for k, v in result.items() if k not in ("takes", "real")}, "out": str(out)},
                           ensure_ascii=False) + "\n")
    log(f"{out.name}: {format_summary(summary)}  -> {lb}")
    return result


def format_summary(s: dict) -> str:
    def f(x, fmt=".3f"):
        return "?" if x is None else format(x, fmt)

    r, ref, rel = s["raw"], s["real"], s["relative"]
    return (f"sim {f(r['sim'])} ({f(rel['sim'], '.0%')} of real {f(ref['sim'])})  "
            f"CER {f(r['cer'], '.2f')} (real {f(ref['cer'], '.2f')})  "
            f"len {f(r['len_ratio'], '.2f')} (real {f(ref['len_ratio'], '.2f')})  stopped {f(r['stopped'], '.0%')}")
