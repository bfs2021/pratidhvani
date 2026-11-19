"""Evaluate synthesized samples: voice similarity, intelligibility, best-take selection.

Similarity is ECAPA cosine against a centroid of the speaker's real clips;
real held-out clips are scored the same way so the numbers have a baseline.
Intelligibility is character error (CER) of the MMS CTC recogniser's greedy
decode against the romanized target. On muffled archival audio CER is high
even for real speech, so compare synthesized against real, never against 0.
"""
import random
import shutil
from pathlib import Path

import numpy as np

from voiceclone.audio import TTS_SR, load16, load_wav
from voiceclone.jsonl import read_jsonl
from voiceclone.text import cer, romanize, syllables


def sample_texts(sample_dir, sentences: list[str] | None = None) -> dict:
    """file -> target text. log.json (written by `synth`) wins; otherwise the
    sample index selects a line of `sentences`."""
    import json

    d = Path(sample_dir)
    log = {e["file"]: e["text"] for e in json.load(open(d / "log.json", encoding="utf-8"))} \
        if (d / "log.json").exists() else {}
    out = {}
    for f in sorted(d.glob("sample_*.wav")):
        if f.name in log:
            out[f.name] = log[f.name]
            continue
        i = int(f.stem.split("_")[1])
        if not sentences or i >= len(sentences):
            raise ValueError(f"{f}: no log.json entry and no sentence #{i}; pass the file you synthesised from")
        out[f.name] = sentences[i]
    return out


def similarity_report(encoder, clips_dir, sample_dirs=(), files=(), n_enroll: int = 40, log=print) -> dict:
    import librosa

    def embed(path):   # 24 kHz first, then 16 kHz: matches how results were reported
        return encoder.embed(librosa.resample(load_wav(path, TTS_SR), orig_sr=TTS_SR, target_sr=16000))

    clips = read_jsonl(Path(clips_dir) / "metadata.jsonl")
    random.Random(0).shuffle(clips)
    enroll = np.stack([embed(Path(clips_dir) / c["file"]) for c in clips[:n_enroll]])
    cent = enroll.mean(0); cent /= np.linalg.norm(cent)
    real = [float(np.dot(embed(Path(clips_dir) / c["file"]), cent)) for c in clips[n_enroll:n_enroll + 20]]
    log(f"real-vs-enrollment (same speaker, held-out clips): mean {np.mean(real):.3f}  min {np.min(real):.3f}")
    report = {"real_same_speaker": {"mean": float(np.mean(real)), "min": float(np.min(real))}}
    for d in sample_dirs:
        scores = {f.name: float(np.dot(embed(f), cent)) for f in sorted(Path(d).glob("sample_*.wav"))}
        log(f"{d}: mean {np.mean(list(scores.values())):.3f}  " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()))
        report[str(d)] = scores
    for f in files:
        report[str(f)] = float(np.dot(embed(f), cent))
        log(f"{f}: {report[str(f)]:.3f}")
    return report


def intelligibility_report(aligner, clips_dir, sentences: list[str] | None, sample_dirs=(), holdout=None,
                           n_real: int = 10, log=print) -> dict:
    import json

    clips = read_jsonl(Path(clips_dir) / "metadata.jsonl")
    real = json.load(open(holdout)) if holdout and Path(holdout).exists() else clips[:n_real]
    rc = [cer(aligner.greedy_decode(load16(Path(clips_dir) / c["file"])), romanize(c["text"])) for c in real[:n_real]]
    log(f"real held-out clips: CER mean {np.mean(rc):.2f}")
    report = {"real": float(np.mean(rc))}
    for d in sample_dirs:
        res = {}
        for name, text in sample_texts(d, sentences).items():
            h = aligner.greedy_decode(load16(Path(d) / name))
            res[name] = cer(h, romanize(text))
            log(f"  {name}: CER {res[name]:.2f} | heard '{h[:50]}' | want '{romanize(text)[:50]}'")
        log(f"{d}: CER mean {np.mean(list(res.values())):.2f}")
        report[str(d)] = res
    return report


def select_take(cands: list[dict], pick: str = "gated") -> dict:
    """Best take for one sentence.

    gated: among takes that are recognisable and plausibly long ("ok"), the one
    closest to the voiceprint; if none is ok, the clearest (lowest CER).
    voice: closest to the voiceprint, ignoring clarity.
    """
    ok = [c for c in cands if c["ok"]] if pick == "gated" else cands
    return max(ok, key=lambda c: c["sim"]) if ok else min(cands, key=lambda c: c["cer"])


def best_of(encoder, aligner, speaker_centroid, take_dirs, out_dir, sentences: list[str] | None = None,
            pick: str = "gated", max_cer: float = 0.85, natural_sps: float = 3.2, log=print) -> list[dict]:
    """Copy the best take of each sentence into out_dir.

    Takes are grouped by the text they were synthesised from (their log.json,
    or `sentences` by index when a folder has no log), so folders made from
    differently ordered sentence files still compare like with like. A take
    is "ok" if its CER is at most max_cer and its length is 0.6-1.5x the
    length expected at natural_sps; if the syllables cannot be counted, the
    length check is skipped.
    """
    import json

    groups: dict[str, list[Path]] = {}
    for d in take_dirs:
        for name, text in sample_texts(d, sentences).items():
            groups.setdefault(text, []).append(Path(d) / name)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    report, rows = [], []
    for i, (text, files) in enumerate(groups.items()):
        expected = syllables(text) / natural_sps
        cands = []
        for f in files:
            y = load16(f)
            c = {"take": str(f.parent), "file": f.name, "sim": float(np.dot(encoder.embed(y), speaker_centroid)),
                 "cer": float(cer(aligner.greedy_decode(y), romanize(text))),
                 "len_ratio": float(len(y) / 16000 / expected) if expected else None}
            plausible = c["len_ratio"] is None or 0.6 <= c["len_ratio"] <= 1.5
            c["ok"] = bool(c["cer"] <= max_cer and plausible)
            cands.append(c)
        best = select_take(cands, pick)
        name = f"sample_{i:02d}.wav"
        shutil.copy(Path(best["take"]) / best["file"], out / name)
        rows.append({"file": name, "text": text})
        report.append({"sentence": text, "best": best, "candidates": cands})
        log(f"[{i}] " + "  ".join(f"{Path(c['take']).name}: sim {c['sim']:.2f} cer {c['cer']:.2f} "
                                  f"len {'?' if c['len_ratio'] is None else format(c['len_ratio'], '.2f')}"
                                  f"{'' if c['ok'] else ' X'}" for c in cands)
            + f"  -> {Path(best['take']).name}")
    with open(out / "log.json", "w", encoding="utf-8") as fh:   # so trim/eval read the right texts
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    if report:
        sims = [r["best"]["sim"] for r in report]; cers = [r["best"]["cer"] for r in report]
        log(f"best-of: voice similarity mean {np.mean(sims):.3f} (min {min(sims):.2f}, max {max(sims):.2f}), "
            f"CER mean {np.mean(cers):.2f}")
    return report
