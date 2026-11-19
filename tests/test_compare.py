import random

import pytest

from voiceclone.compare import compare_runs, listening_results, paired_differences, verdict

SENTENCES = [f"ವಾಕ್ಯ {i}" for i in range(8)]


def run(sim, cer=lambda s, k: 0.5, name="a", spec="h1", shuffle=None):
    """A scores.json-like run: sim(sentence_index, seed) per take."""
    takes = [{"text": s, "seed": k, "sim": sim(i, k), "cer": cer(i, k), "len_err": 0.1, "stopped": True}
             for i, s in enumerate(SENTENCES) for k in (0, 1, 2)]
    if shuffle is not None:
        random.Random(shuffle).shuffle(takes)
    return {"run": name, "spec_hash": spec, "takes": takes, "real": []}


def varied(i, k):
    """Similarity that differs a lot between sentences and seeds, like real takes do."""
    return 0.4 + 0.05 * i + 0.03 * ((i * 7 + k * 3) % 5)


def test_identical_runs_show_no_difference():
    res = compare_runs(run(varied), run(varied, name="b"))
    assert all(r["verdict"] == "no clear difference" for r in res.values())
    assert res["sim"]["diff"] == 0 and res["sim"]["ci"] == [0, 0]


def test_takes_are_paired_by_sentence_and_seed_not_by_position():
    # B is A plus 0.02 on every take, but lists its takes in another order;
    # positional pairing would compare unrelated takes and see mostly noise
    a, b = run(varied), run(lambda i, k: varied(i, k) + 0.02, name="b", shuffle=1)
    diffs = [d for g in paired_differences(a, b, "sim").values() for d in g]
    assert len(diffs) == len(SENTENCES) * 3 and diffs == pytest.approx([0.02] * len(diffs))
    assert compare_runs(a, b)["sim"]["verdict"] == "B better"


def test_seed_noise_is_not_reported_as_an_improvement():
    # B is better on half the sentences and as much worse on the others, with seed jitter
    noise = lambda i, k: (0.04 if i % 2 else -0.04) + 0.01 * (k - 1)
    res = compare_runs(run(varied), run(lambda i, k: varied(i, k) + noise(i, k), name="b"))
    assert res["sim"]["verdict"] == "no clear difference"
    assert res["sim"]["ci"][0] < 0 < res["sim"]["ci"][1]


def test_a_few_good_sentences_do_not_make_a_better_run():
    # a big gain on two of eight sentences: resampling whole sentences keeps the interval honest
    # (resampling the 24 takes one by one would call this "B better")
    res = compare_runs(run(varied), run(lambda i, k: varied(i, k) + (0.3 if i < 2 else 0), name="b"))
    assert res["sim"]["verdict"] == "no clear difference"


def test_lower_cer_is_better():
    res = compare_runs(run(varied), run(varied, cer=lambda i, k: 0.4, name="b"))
    assert res["cer"]["verdict"] == "B better"
    assert verdict(-0.2, -0.1, higher_is_better=False) == "B better"


def test_runs_of_different_specs_are_refused():
    with pytest.raises(ValueError, match="different benchmark specs"):
        compare_runs(run(varied), run(varied, name="b", spec="h2"))


def test_incomplete_run_is_refused():
    b = run(varied, name="b"); b["takes"].pop()
    with pytest.raises(ValueError, match="different takes"):
        compare_runs(run(varied), b)


# --- listening answers ---------------------------------------------------
def key():
    trials = {"t00": {"kind": "pair", "1": "A", "2": "B"}, "t01": {"kind": "pair", "1": "B", "2": "A"},
              "t02": {"kind": "pair", "1": "A", "2": "B"}, "t03": {"kind": "check", "1": "real", "2": "B"}}
    return {"page_id": "p1", "run_a": "runs/a", "run_b": "runs/b", "trials": trials}


def answers(voice, natural, listener="x"):
    return {"page_id": "p1", "listener": listener,
            "answers": [{"trial": t, "voice": v, "natural": n} for t, v, n in zip(sorted(key()["trials"]), voice, natural)]}


def test_listening_win_rate_follows_the_hidden_side_not_the_number():
    # always picking B, wherever it was placed, is a 100% B win; picking "1" every time is not
    res = listening_results(key(), [answers(["2", "1", "2", "1"], ["2", "1", "2", "1"])], "runs/a", "runs/b")
    assert res["questions"]["voice"]["b_win_rate"] == 1.0
    assert res["listeners"][0]["picked_real"]["voice"] == "1/1"
    res = listening_results(key(), [answers(["1", "1", "1", "2"], ["same"] * 4)], "runs/a", "runs/b")
    assert res["questions"]["voice"]["b_win_rate"] == pytest.approx(1 / 3)
    assert res["questions"]["natural"]["tie"] == 3 and res["questions"]["natural"]["b_win_rate"] is None


def test_listening_runs_can_be_named_in_either_order():
    ans = [answers(["2", "1", "2", "1"], ["2", "1", "2", "1"])]
    assert listening_results(key(), ans, "runs/b", "runs/a")["questions"]["voice"]["b_win_rate"] == 0.0


def test_three_listening_wins_are_not_a_preference():
    res = listening_results(key(), [answers(["2", "1", "2", "1"], ["2", "1", "2", "1"])], "runs/a", "runs/b")
    assert res["questions"]["voice"]["verdict"] == "no clear preference"


def test_answers_for_another_page_are_refused():
    bad = answers(["1"] * 4, ["1"] * 4); bad["page_id"] = "other"
    with pytest.raises(ValueError, match="page"):
        listening_results(key(), [bad], "runs/a", "runs/b")
