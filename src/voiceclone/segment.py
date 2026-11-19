"""Cut recordings into training-sized chunks.

`vad_segment`: transcript-free, cuts at natural pauses (energy threshold).
`segment_aligned`: cuts force-aligned speech of one target speaker.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceclone.audio import peak_normalize, write_wav
from voiceclone.jsonl import write_jsonl
from voiceclone.text import strip_quotes, syllables


@dataclass
class VadParams:
    frame: float = 0.02        # analysis frame, s
    min_pause: float = 0.25    # a pause shorter than this is not a cut point
    min_dur: float = 3.0
    max_dur: float = 10.0
    ideal: float = 8.0         # merge pause-delimited pieces up to about this long
    pad: float = 0.1
    above_floor_db: float = 12.0   # pause threshold = 10th-percentile level + this


def vad_cuts(wav: np.ndarray, sr: int, p: VadParams = VadParams()):
    """Chunk boundaries (s) and the pause threshold (dB) for a mono waveform."""
    hop = int(p.frame * sr)
    n = len(wav) // hop
    e = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-8
    db = 20 * np.log10(np.convolve(e, np.ones(5) / 5, "same"))
    th = np.percentile(db, 10) + p.above_floor_db
    quiet = db < th

    cuts, i = [0.0], 0
    while i < n:                                   # middle of each long-enough pause
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * p.frame >= p.min_pause:
                cuts.append((i + j) / 2 * p.frame)
            i = j
        else:
            i += 1
    cuts.append(n * p.frame)

    segs, s = [], cuts[0]                          # greedy merge into chunks
    for a, b in zip(cuts, cuts[1:]):
        if b - s > p.ideal and a - s >= p.min_dur:
            segs.append((s, a)); s = a
    segs.append((s, cuts[-1]))

    def split_long(a, b):
        """Split a span longer than max_dur at its quietest frame, away from the edges."""
        if b - a <= p.max_dur:
            return [(a, b)]
        fa, fb = int((a + p.min_dur) / p.frame), int((b - p.min_dur) / p.frame)
        m = (fa + int(np.argmin(db[fa:fb]))) * p.frame
        return split_long(a, m) + split_long(m, b)

    return [q for a, b in segs for q in split_long(a, b)], th, len(cuts) - 2


def vad_segment(wav: np.ndarray, sr: int, out_dir, prefix: str = "seg", p: VadParams = VadParams()) -> dict:
    """Write chunk WAVs + segments.jsonl {file,start,end,duration}; returns a summary."""
    segs, th, n_pauses = vad_cuts(wav, sr, p)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob(f"{prefix}_*.wav"):
        f.unlink()
    rows, dropped = [], 0
    for a, b in segs:
        if not (p.min_dur <= b - a <= p.max_dur + 0.5):
            dropped += 1; continue
        x = wav[int(max(0, a - p.pad) * sr): int((b + p.pad) * sr)]
        k = np.flatnonzero(np.abs(x) > 10 ** ((th + 6) / 20))   # trim silence inside the chunk
        if len(k) == 0:
            dropped += 1; continue
        x = x[max(0, k[0] - int(0.1 * sr)): k[-1] + int(0.15 * sr)]
        if len(x) / sr < p.min_dur:
            dropped += 1; continue
        name = f"{prefix}_{len(rows):04d}.wav"
        write_wav(out / name, peak_normalize(x), sr)
        rows.append({"file": name, "start": round(a, 2), "end": round(b, 2), "duration": round(len(x) / sr, 2)})
    write_jsonl(out / "segments.jsonl", rows)
    return {"threshold_db": float(th), "pauses": n_pauses, "chunks": len(rows),
            "minutes": sum(r["duration"] for r in rows) / 60, "dropped": dropped}


@dataclass
class AlignedSegmentParams:
    min_dur: float = 2.0
    max_dur: float = 11.0
    ideal: float = 10.0
    pause: float = 0.35        # a gap longer than this between words is a break point
    lead: float = 0.12
    tail: float = 0.20
    guard: float = 0.5         # keep this far from any other speaker's words
    min_sps: float = 2.0       # syllables/s; slower means untranscribed speech
    min_mean: float = -1.8     # mean word score
    min_word: float = -4.0     # worst word score
    max_word_dur: float = 1.8


BREAK = re.compile(r"[.?!,;:।॥…]['\"]?$")


def segment_aligned(aligned: list[dict], wav: np.ndarray, sr: int, out_dir, target: str,
                    p: AlignedSegmentParams = AlignedSegmentParams()) -> dict:
    """Cut force-aligned speech of `target` into clips + metadata.jsonl.

    Break points are clause punctuation or pauses; units are merged up to
    `ideal` seconds. A clip is kept only if it is clear of other speakers,
    its text aligns well, no word is implausibly long, and the rate is
    plausible.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("clip_*.wav"):
        f.unlink()
    others = [(w["start"] - p.guard, w["end"] + p.guard)
              for u in aligned if u["speaker"] != target for w in u["words"]]

    cands = []
    for u in aligned:
        if u["speaker"] != target or not u["words"]:
            continue
        ws, units, cur = u["words"], [], []
        for i, w in enumerate(ws):
            cur.append(w)
            gap = ws[i + 1]["start"] - w["end"] if i + 1 < len(ws) else 9
            if BREAK.search(w["text"]) or gap > p.pause:
                units.append(cur); cur = []
        if cur:
            units.append(cur)
        group = []
        for un in units:
            if group and un[-1]["end"] - group[0]["start"] > p.ideal:
                cands.append(group); group = []
            group = group + un
        if group:
            cands.append(group)

    clips, why, last_end = [], {}, -1.0

    def drop(r):
        why[r] = why.get(r, 0) + 1

    for g in cands:
        s, e = g[0]["start"] - p.lead, g[-1]["end"] + p.tail
        sc = [w["score"] for w in g]
        if e - s < p.min_dur: drop("short"); continue
        if e - s > p.max_dur: drop("long"); continue
        if np.mean(sc) < p.min_mean or min(sc) < p.min_word: drop("score"); continue
        if max(w["end"] - w["start"] for w in g) > p.max_word_dur: drop("long_word"); continue
        if syllables(" ".join(w["text"] for w in g)) / (e - s) < p.min_sps: drop("slow"); continue
        if any(a < e and s < b for a, b in others): drop("other_speaker"); continue
        s = max(s, last_end)                      # never share audio between clips
        if e - s < p.min_dur: drop("overlap"); continue
        name = f"clip_{len(clips):04d}.wav"
        write_wav(out / name, peak_normalize(wav[int(s * sr):int(e * sr)]), sr)
        clips.append({"file": name, "text": strip_quotes(" ".join(w["text"] for w in g)), "start": round(s, 3),
                      "end": round(e, 3), "duration": round(e - s, 3), "score": round(float(np.mean(sc)), 3)})
        last_end = e
    write_jsonl(out / "metadata.jsonl", clips)
    return {"candidates": len(cands), "kept": len(clips),
            "minutes": sum(c["duration"] for c in clips) / 60, "dropped": why}
