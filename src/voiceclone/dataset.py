"""Build the fine-tuning clip set from transcribed chunks."""
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceclone.audio import load16
from voiceclone.jsonl import read_jsonl, write_jsonl
from voiceclone.text import collapse_spaces, syllables


@dataclass
class ClipFilter:
    min_sim: float = 0.5        # ECAPA similarity to the speaker; other voices score near 0
    min_mean: float = -1.6      # mean per-frame CTC score of the transcript's words
    min_word: float = -4.0
    max_word_dur: float = 1.8
    min_rate: float = 2.0       # syllables/s
    max_rate: float = 9.0
    min_words: int = 3
    min_f0: float | None = None  # median pitch window (Hz); None = no pitch gate.
    max_f0: float | None = None  # catches singers, chanting and instruments that pass the voiceprint


F0_FLOOR, F0_CEIL = 60.0, 500.0   # pYIN search range: covers adult speech and the singing it must reject


def median_f0(y16: np.ndarray, sr: int = 16000) -> float | None:
    """Median pitch of the voiced frames (pYIN), or None if nothing is voiced."""
    import librosa

    f0, voiced, _ = librosa.pyin(y16, fmin=F0_FLOOR, fmax=F0_CEIL, sr=sr)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else None


def f0_range(clip_paths, lo_pct: float = 5, hi_pct: float = 95) -> tuple[float, float]:
    """Pitch window from clips that are certainly the speaker: the 5th to 95th
    percentile of their median pitch. A fixed window (e.g. 135-195 Hz) drops
    the speaker's own raised, emphatic speech; his own clips set the limits."""
    meds = [m for m in (median_f0(load16(p)) for p in clip_paths) if m is not None]
    if len(meds) < 5:
        raise ValueError(f"only {len(meds)} enrollment clips have voiced speech; need 5 to set a pitch window")
    lo, hi = np.percentile(meds, [lo_pct, hi_pct])
    return float(lo), float(hi)


def build_clips(vad_dirs, out_dir, encoder, aligner, speaker_centroid,
                f: ClipFilter = ClipFilter(), log=print) -> dict:
    """Keep chunks that are the target speaker and whose transcript fits the audio.

    Each vad dir holds chunk WAVs, segments.jsonl and transcripts.jsonl.
    Writes the kept WAVs + metadata.jsonl {file,text,start,end,duration,score,sim[,f0][,source]};
    `source` (the recording, e.g. from import-vault) is kept so benchmarks can tell what trained.
    """
    segs, trans = {}, []
    for d in map(Path, vad_dirs):
        for r in read_jsonl(d / "segments.jsonl"):
            segs[r["file"]] = {**r, "dir": d}
        trans += read_jsonl(d / "transcripts.jsonl")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.wav"):
        p.unlink()
    kept, why = [], {}

    def drop(r):
        why[r] = why.get(r, 0) + 1

    for t in trans:
        s, text = segs[t["file"]], collapse_spaces(t["text"])
        y16 = load16(s["dir"] / t["file"])
        if len(text.split()) < f.min_words: drop("few_words"); continue
        sim = float(np.dot(encoder.embed(y16), speaker_centroid))
        if sim < f.min_sim: drop("other_speaker"); continue
        f0 = None
        if f.min_f0 is not None or f.max_f0 is not None:
            f0 = median_f0(y16)
            if f0 is None or not (f.min_f0 or 0) <= f0 <= (f.max_f0 or np.inf): drop("pitch"); continue
        rate = syllables(text) / s["duration"]
        if not f.min_rate <= rate <= f.max_rate: drop("rate"); continue
        ws = aligner.align_text(y16, text)
        if not ws: drop("align"); continue
        pf = [w["pf"] for w in ws]
        if np.mean(pf) < f.min_mean or min(pf) < f.min_word: drop("score"); continue
        if max(w["end"] - w["start"] for w in ws) > f.max_word_dur: drop("long_word"); continue
        shutil.copy(s["dir"] / t["file"], out / t["file"])
        kept.append({"file": t["file"], "text": text, "start": s["start"], "end": s["end"],
                     "duration": s["duration"], "score": round(float(np.mean(pf)), 3), "sim": round(sim, 3),
                     **({"f0": round(f0, 1)} if f0 is not None else {}),
                     **({"source": s["source"]} if s.get("source") else {})})
    write_jsonl(out / "metadata.jsonl", kept)
    return {"transcribed": len(trans), "kept": len(kept),
            "minutes": sum(k["duration"] for k in kept) / 60, "dropped": why}
