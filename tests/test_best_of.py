import json

import numpy as np
import pytest
import soundfile as sf

from voiceclone.evaluate import best_of, sample_texts, select_take


def c(take, sim, cer, ok):
    return {"take": take, "sim": sim, "cer": cer, "ok": ok}


def test_gated_prefers_voice_match_among_clear_takes():
    # the closest voice is garbled (not ok), so a clear take wins
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.60, True), c("d", 0.65, 0.50, True)]
    assert select_take(cands, "gated")["take"] == "b"


def test_gated_falls_back_to_clearest_when_nothing_passes():
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.90, False)]
    assert select_take(cands, "gated")["take"] == "b"


def test_voice_mode_ignores_clarity():
    cands = [c("a", 0.80, 0.95, False), c("b", 0.70, 0.60, True)]
    assert select_take(cands, "voice")["take"] == "a"


# --- pairing takes with the sentence they actually say -------------------
class FakeEncoder:
    """Voiceprint = the clip's amplitude, so a louder take is 'more like him'."""
    def embed(self, y):
        return np.array([float(np.abs(y).max())])


class FakeAligner:
    def greedy_decode(self, y):
        return ""


def write_take(d, rows, amps):
    d.mkdir()
    for (name, _), amp in zip(rows, amps):
        sf.write(d / name, amp * np.ones(16000 * 3, dtype=np.float32), 16000)
    json.dump([{"file": n, "text": t} for n, t in rows], open(d / "log.json", "w", encoding="utf-8"),
              ensure_ascii=False)


def test_takes_are_paired_by_logged_text_not_by_position(tmp_path):
    s1, s2 = "ನಮಸ್ಕಾರ ರೀ", "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ"
    write_take(tmp_path / "a", [("sample_00.wav", s1), ("sample_01.wav", s2)], [0.2, 0.2])
    # b was synthesised from the same sentences in the opposite order
    write_take(tmp_path / "b", [("sample_00.wav", s2), ("sample_01.wav", s1)], [0.9, 0.1])
    out = tmp_path / "best"
    rep = best_of(FakeEncoder(), FakeAligner(), np.array([1.0]), [tmp_path / "a", tmp_path / "b"], out,
                  pick="voice")
    by_text = {r["sentence"]: r for r in rep}
    assert len(rep) == 2 and all(len(r["candidates"]) == 2 for r in rep)
    assert by_text[s2]["best"]["take"].endswith("b")   # b's louder take of s2 wins s2
    assert by_text[s1]["best"]["take"].endswith("a")   # b's take of s1 is quieter
    # the output folder records which sentence each picked file says
    assert set(sample_texts(out).values()) == {s1, s2}


def test_unreadable_script_skips_the_length_check_instead_of_crashing(tmp_path):
    write_take(tmp_path / "a", [("sample_00.wav", "سلام دنیا")], [0.5])
    rep = best_of(FakeEncoder(), FakeAligner(), np.array([1.0]), [tmp_path / "a"], tmp_path / "best")
    assert rep[0]["best"]["len_ratio"] is None


def test_folder_without_log_needs_the_sentence_file(tmp_path):
    (tmp_path / "a").mkdir()
    sf.write(tmp_path / "a" / "sample_00.wav", np.zeros(1600, np.float32), 16000)
    with pytest.raises(ValueError, match="no log.json"):
        sample_texts(tmp_path / "a")
    assert sample_texts(tmp_path / "a", ["ಒಂದು"]) == {"sample_00.wav": "ಒಂದು"}
