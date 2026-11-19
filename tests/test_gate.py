"""The gate decides what reaches main, so its rules are tested as rules:
a change that makes any measured part of the voice worse with confidence must
not merge; an experiment must show an improvement, not just noise; a run that
trained on the test recordings can never pass."""
import pytest

from voiceclone.compare import compare_runs
from voiceclone.gate import evaluate_gate, format_markdown
from voiceclone.recipe import RecipeError, load_recipe, run_recipe

SENTENCES = [f"ವಾಕ್ಯ {i}" for i in range(12)]


def run(name, sim, cer, stopped=1, leak=None):
    takes = [{"text": s, "seed": k, "sim": sim + 0.01 * ((i + k) % 3 - 1), "cer": cer + 0.01 * ((i * k) % 3 - 1),
              "len_err": 0.1, "stopped": stopped} for i, s in enumerate(SENTENCES) for k in range(3)]
    return {"run": name, "spec": "kn-north-v1", "spec_hash": "abc", "takes": takes, "leak": leak,
            "summary": {"real": {"sim": 0.76}}}


def gate(a, b, **kw):
    m = compare_runs(a, b, n_boot=2000)
    return evaluate_gate(a, b, m, **kw), m


def test_clear_regression_fails():
    res, _ = gate(run("main", 0.65, 0.40), run("pr", 0.58, 0.40))
    assert not res["passed"] and res["worse"] == ["sim"]


def test_worse_cer_fails_even_if_similarity_improves():
    # a voice that sounds closer but says the wrong words is not better
    res, _ = gate(run("main", 0.60, 0.30), run("pr", 0.66, 0.45))
    assert not res["passed"] and "cer" in res["worse"] and "sim" in res["better"]


def test_noise_level_difference_is_neutral_and_passes_for_code_changes():
    res, _ = gate(run("main", 0.62, 0.40), run("pr", 0.62, 0.40))
    assert res["passed"] and res["outcome"] == "neutral"


def test_experiment_must_show_an_improvement():
    res, _ = gate(run("main", 0.62, 0.40), run("pr", 0.62, 0.40), require_improvement=True)
    assert not res["passed"]
    res, _ = gate(run("main", 0.62, 0.40), run("pr", 0.68, 0.40), require_improvement=True)
    assert res["passed"] and res["outcome"] == "improved"


def test_leaked_run_never_passes():
    res, _ = gate(run("main", 0.60, 0.40), run("pr", 0.75, 0.30, leak={"allowed": True, "leaked": ["K06"]}))
    assert not res["passed"] and "allow-leak" in res["reasons"][0]


def test_report_names_the_verdict_and_ceiling():
    a, b = run("main", 0.65, 0.40), run("pr", 0.58, 0.40)
    res, m = gate(a, b)
    md = format_markdown(a, b, m, res)
    assert md.startswith("### Quality gate: FAIL")
    assert "speaker similarity is worse than main" in md and "0.760" in md


# ---------- recipes ----------
def write(tmp_path, recipe):
    import json

    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(recipe))
    return p


TRAIN = ["train", "--clips", "{data}/clips_v2", "--out", "{work}/adapter", "--epochs", "3"]


def test_recipe_runs_steps_then_benchmarks_the_adapter(tmp_path):
    calls = []
    p = write(tmp_path, {"name": "v2", "steps": [TRAIN], "adapter": "{work}/adapter"})
    out = run_recipe(p, "/data", tmp_path / "w", "/data/bench.json", python="py",
                     run=lambda argv, check: calls.append(argv), log=lambda *a: None)
    assert calls[0] == ["py", "-m", "voiceclone.cli", "train", "--clips", "/data/clips_v2",
                        "--out", str(tmp_path / "w/adapter"), "--epochs", "3"]
    assert calls[1][3:] == ["benchmark", "--spec", "/data/bench.json", "--out", str(tmp_path / "w/bench"),
                            "--lora", str(tmp_path / "w/adapter")]
    assert out == tmp_path / "w" / "bench"


@pytest.mark.parametrize("step,why", [
    (["benchmark", "--spec", "x", "--out", "{work}/b", "--allow-leak"], "not a recipe step"),   # the gate benchmarks, never the recipe
    (["serve"], "not a recipe step"),
    (["train", "--clips", "{work}/c", "--out", "{data}/adapter"], "under {work}"),               # the shared data is read-only
    (["train", "--clips", "{work}/c"], "must be given"),                                         # default --out lands outside the run
    (["transcribe", "--vad", "{data}/vad_K06"], "under {work}"),                                  # transcribe writes in place
    (["train", "--clips", "{work}/c", "--out", "{work}/../../x"], "under {work}"),
    (["train", "--clips", "{work}/c", "--out={work}/a"], "two strings"),
])
def test_recipe_refuses_steps_that_could_corrupt_data_or_the_score(tmp_path, step, why):
    with pytest.raises(RecipeError, match=why.replace("{", r"\{").replace("}", r"\}")):
        load_recipe(write(tmp_path, {"steps": [step], "adapter": None}))
