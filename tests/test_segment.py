"""Chunks must start and end inside pauses, or training clips begin mid-word."""
import numpy as np

from voiceclone.jsonl import read_jsonl
from voiceclone.segment import VadParams, vad_cuts, vad_segment

SR = 16000


def speech_with_pauses(words=12, word_s=1.4, pause_s=0.6, seed=0):
    rng = np.random.default_rng(seed)
    parts, spans, t = [], [], 0.0
    for _ in range(words):
        parts.append(0.3 * rng.standard_normal(int(word_s * SR)).astype(np.float32)); spans.append((t, t + word_s))
        t += word_s
        parts.append(1e-4 * rng.standard_normal(int(pause_s * SR)).astype(np.float32))
        t += pause_s
    return np.concatenate(parts), spans


def test_cuts_fall_inside_pauses_and_respect_length_limits():
    wav, spans = speech_with_pauses()
    segs, _, n_pauses = vad_cuts(wav, SR)
    assert n_pauses >= 10
    for a, b in segs[:-1]:
        assert not any(s + 0.05 < b < e - 0.05 for s, e in spans), f"cut at {b:.2f}s is inside speech"
    assert all(b - a <= VadParams().max_dur for a, b in segs)


def test_vad_segment_writes_chunks_and_index(tmp_path):
    wav, _ = speech_with_pauses()
    summary = vad_segment(wav, SR, tmp_path, prefix="t")
    rows = read_jsonl(tmp_path / "segments.jsonl")
    assert summary["chunks"] == len(rows) > 0
    assert all((tmp_path / r["file"]).exists() for r in rows)
    assert all(3.0 <= r["duration"] <= 10.5 + 0.3 for r in rows)
