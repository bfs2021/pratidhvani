"""The pitch gate (from EXP-004/005): recordings of talks also hold chanting,
singing and other voices that can pass the voiceprint filter, and training on
them washed the speaker's identity out. The gate must reject those, keep the
speaker's own range, and be off unless asked for."""
import json

import numpy as np
import pytest
import soundfile as sf

from voiceclone.dataset import ClipFilter, build_clips, f0_range, median_f0

SR = 16000


def voiced(f0, sec=3.0):
    """A harmonic tone with a little vibrato: pYIN treats it as voiced speech."""
    t = np.arange(int(sec * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))) / SR
    return (0.3 * sum(np.sin(k * phase) / k for k in range(1, 6))).astype(np.float32)


@pytest.mark.parametrize("f0", [110, 160, 280])
def test_median_f0(f0):
    assert median_f0(voiced(f0)) == pytest.approx(f0, rel=0.03)


def test_silence_has_no_pitch():
    assert median_f0(np.zeros(SR * 2, np.float32)) is None


def test_window_comes_from_the_speakers_own_clips(tmp_path):
    paths = []
    for i, f0 in enumerate([140, 150, 155, 160, 170, 175, 190]):
        p = tmp_path / f"e{i}.wav"; sf.write(p, voiced(f0), SR); paths.append(p)
    lo, hi = f0_range(paths)
    assert 140 <= lo < 150 and 180 < hi <= 192


class Encoder:
    def embed(self, y):
        return np.array([1.0])


class Aligner:
    def align_text(self, y, text):
        return [{"pf": -0.5, "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(len(text.split()))]


def chunks(tmp_path, pitches):
    d = tmp_path / "vad"; d.mkdir()
    segs, trans = [], []
    for i, f0 in enumerate(pitches):
        f = f"c_{i:04d}.wav"
        sf.write(d / f, voiced(f0), SR)
        segs.append({"file": f, "start": 0, "end": 3, "duration": 3.0})
        trans.append({"file": f, "text": "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ ಆರಾಮ ಇದ್ದೀರಿ"})
    (d / "segments.jsonl").write_text("".join(json.dumps(s) + "\n" for s in segs))
    (d / "transcripts.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trans))
    return d


def test_gate_drops_chanting_and_keeps_the_speaker(tmp_path):
    d = chunks(tmp_path, [150, 165, 290, 420])
    res = build_clips([d], tmp_path / "out", Encoder(), Aligner(), np.array([1.0]),
                      ClipFilter(min_f0=130, max_f0=200), log=lambda *a: None)
    assert res["kept"] == 2 and res["dropped"] == {"pitch": 2}
    meta = [json.loads(line) for line in open(tmp_path / "out" / "metadata.jsonl")]
    assert [round(m["f0"] / 10) for m in meta] == [15, 16]


def test_gate_is_off_by_default(tmp_path):
    d = chunks(tmp_path, [150, 420])
    res = build_clips([d], tmp_path / "out", Encoder(), Aligner(), np.array([1.0]), log=lambda *a: None)
    assert res["kept"] == 2
    assert all("f0" not in json.loads(line) for line in open(tmp_path / "out" / "metadata.jsonl"))
