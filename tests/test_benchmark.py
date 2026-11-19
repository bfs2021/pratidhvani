import json

import numpy as np
import pytest
import soundfile as sf

from voiceclone.benchmark import (LeakError, holdout_clips, leak_guard, load_spec, score_real, source_of,
                                  spec_hash, synthesize_takes)
from voiceclone.svara.synth import GenParams
from voiceclone.svara.tokens import token_budget
from voiceclone.text import syllables


def clips_dir(path, files, sources=None):
    path.mkdir(parents=True)
    rows = [{"file": f, "text": "ನಮಸ್ಕಾರ ರೀ", "duration": 3.0, **({"source": s} if s else {})}
            for f, s in zip(files, sources or [None] * len(files))]
    (path / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for f in files:
        sf.write(path / f, np.zeros(1600, np.float32), 16000)
    return path


def adapter(path, clips, init_adapter=None):
    path.mkdir(parents=True)
    (path / "train_log.json").write_text(json.dumps(
        {"steps": 10, "clips": str(clips), "params": {"init_adapter": str(init_adapter) if init_adapter else None}}))
    return path


@pytest.fixture
def spec(tmp_path):
    hold = clips_dir(tmp_path / "hold", ["K06_0000.wav", "K06_0001.wav", "K06_0002.wav", "K14_0000.wav"])
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"holdout": {"clips": str(hold), "sources": ["K06"]},
                             "conditioning": {"wav": str(hold / "K06_0000.wav"), "text": "ನಮಸ್ಕಾರ"},
                             "sentences": ["ನಮಸ್ಕಾರ ರೀ", "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ"]}), encoding="utf-8")
    return load_spec(p)


def test_source_is_the_recording_not_the_chunk():
    assert source_of("K06_0012.wav") == source_of("K06_0999.wav") == "K06"
    assert source_of({"file": "abc123.wav", "source": "talk7"}) == "talk7"   # import-vault keeps the real source


def test_leak_guard_refuses_an_adapter_trained_on_a_benchmark_recording(tmp_path, spec):
    train = clips_dir(tmp_path / "train", ["K06_0100.wav", "K15_0000.wav"])
    with pytest.raises(LeakError, match="K06"):
        leak_guard(spec, adapter(tmp_path / "lora", train))


def test_leak_guard_passes_a_clean_adapter_and_records_its_data(tmp_path, spec):
    train = clips_dir(tmp_path / "train", ["K14_0001.wav", "K15_0000.wav", "K15_0001.wav"])
    info = leak_guard(spec, adapter(tmp_path / "lora", train))
    assert info["leaked"] == [] and info["trained_sources"] == {"K14": 1, "K15": 2}


def test_leak_guard_follows_continued_training(tmp_path, spec):
    # the final adapter only saw K15, but it continued one that trained on K06
    first = adapter(tmp_path / "lora_v1", clips_dir(tmp_path / "c1", ["K06_0100.wav"]))
    second = adapter(tmp_path / "lora_v2", clips_dir(tmp_path / "c2", ["K15_0000.wav"]), init_adapter=first)
    with pytest.raises(LeakError, match="K06"):
        leak_guard(spec, second)


def test_leak_guard_checks_the_conditioning_clip_source(tmp_path, spec):
    spec["conditioning"]["source"] = "K26"
    train = clips_dir(tmp_path / "train", ["K26_0000.wav"])
    with pytest.raises(LeakError, match="K26"):
        leak_guard(spec, adapter(tmp_path / "lora", train))


def test_leak_guard_refuses_when_training_data_is_unknown(tmp_path, spec):
    (tmp_path / "lora").mkdir()
    with pytest.raises(LeakError, match="train_log"):
        leak_guard(spec, tmp_path / "lora")
    info = leak_guard(spec, tmp_path / "lora", allow=True, log=lambda *a: None)
    assert info["allowed"] and info["leaked"] is None   # scored, but flagged as unverified


def test_checkpoint_folders_use_the_run_train_log(tmp_path, spec):
    lora = adapter(tmp_path / "lora", clips_dir(tmp_path / "train", ["K06_0100.wav"]))
    (lora / "checkpoint-00050").mkdir()
    with pytest.raises(LeakError):
        leak_guard(spec, lora / "checkpoint-00050")


def test_allow_leak_scores_but_flags(tmp_path, spec):
    lora = adapter(tmp_path / "lora", clips_dir(tmp_path / "train", ["K06_0100.wav"]))
    info = leak_guard(spec, lora, allow=True, log=lambda *a: None)
    assert info["leaked"] == ["K06"] and info["allowed"]


def test_holdout_is_whole_recordings_without_the_conditioning_clip(spec):
    files = {c["file"] for c in holdout_clips(spec)}
    assert files == {"K06_0001.wav", "K06_0002.wav"}   # not K14, not the conditioning clip


def test_spec_hash_changes_with_any_benchmark_input(spec):
    h = spec_hash(spec)
    spec["seeds"] = [0, 1]
    assert spec_hash(spec) != h


class Onehot:
    """Clip i embeds to the i-th unit vector (by its amplitude), so clips share nothing."""
    def embed(self, y):
        e = np.zeros(3); e[int(round(np.abs(y).max() * 10)) - 1] = 1.0
        return e


class Deaf:
    def greedy_decode(self, y):
        return ""


def test_real_clips_are_not_scored_against_themselves(tmp_path):
    rows = []
    for i in range(3):
        sf.write(tmp_path / f"c{i}.wav", np.full(16000, (i + 1) / 10, np.float32), 16000)
        rows.append({"file": f"c{i}.wav", "path": str(tmp_path / f"c{i}.wav"), "text": "ನಮಸ್ಕಾರ ರೀ"})
    _, real = score_real(Onehot(), Deaf(), rows)
    # with itself in the centroid each clip would score 1/sqrt(3); against the others it shares nothing
    assert all(r["sim"] == pytest.approx(0.0) for r in real)


class FakeSynth:
    """Says every sentence for `n_tokens(text)` tokens; counts calls."""
    def __init__(self, n_tokens):
        self.n_tokens, self.calls = n_tokens, 0

    def reference(self, wav):
        return [0] * 7

    def generate(self, text, ref_tokens, ref_text, seed=0, p=None):
        self.calls += 1
        return np.zeros(2400, np.float32), self.n_tokens(text)


def test_takes_cover_every_sentence_and_seed_and_flag_runaways(tmp_path, spec):
    cap = lambda t: token_budget(syllables(t), GenParams().max_sec, GenParams().min_sps, GenParams().max_sps)[1]
    long_one = spec["sentences"][1]
    synth = FakeSynth(lambda t: cap(t) if t == long_one else 70)
    takes = synthesize_takes(synth, spec, tmp_path / "run", log=lambda *a: None)
    assert {(t["text"], t["seed"]) for t in takes} == {(s, k) for s in spec["sentences"] for k in (0, 1, 2)}
    assert all(t["stopped"] == (t["text"] != long_one) for t in takes)   # ran to the cap = did not stop


def test_interrupted_benchmark_resumes_without_resynthesizing(tmp_path, spec):
    synthesize_takes(FakeSynth(lambda t: 70), spec, tmp_path / "run", log=lambda *a: None)
    again = FakeSynth(lambda t: 70)
    synthesize_takes(again, spec, tmp_path / "run", log=lambda *a: None)
    assert again.calls == 0


def test_a_take_with_no_audio_scores_worst_not_missing(tmp_path):
    from voiceclone.benchmark import score_takes

    sf.write(tmp_path / "empty.wav", np.zeros(800, np.float32), 16000)   # 0.05 s
    t = {"path": str(tmp_path / "empty.wav"), "text": "ನಮಸ್ಕಾರ ರೀ", "seed": 0, "sentence": 0}
    s = score_takes(None, None, np.ones(3), [t], real_len_ratio=1.0, log=lambda *a: None)[0]
    # dropping it instead would let a run look better by failing on hard sentences
    assert s["empty"] and s["sim"] == 0.0 and s["cer"] == 1.0 and s["len_err"] > 2
