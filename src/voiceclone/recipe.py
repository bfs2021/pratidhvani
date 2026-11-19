"""A recipe: the `voiceclone` commands that turn the private data into an
adapter. ci/recipe.json is the recipe main ships; an experiment is a change
to it (or to the code it runs), and the quality gate scores the result.

  {"name": "...",
   "steps": [["build-clips", "--vad", "{data}/vad_K06", ..., "--out", "{work}/clips"],
             ["train", "--clips", "{work}/clips", "--out", "{work}/adapter"]],
   "adapter": "{work}/adapter"}          (null: score the zero-shot model)

{data} is the shared, read-only data folder on the gate runner and {work}
this run's folder. Steps may only write under {work}, and the benchmark is
not a step: the gate runs it itself, so a recipe cannot pass --allow-leak.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from voiceclone.jsonl import write_json

# step -> the flags it writes to; each must be given and point under {work}
# (left out, they default to folders outside the run)
OUTPUTS = {"restore": ["--out"], "parse-transcript": ["--out"], "align": ["--out", "--emissions-cache"],
           "segment": ["--out"], "vad": ["--out"], "transcribe": ["--vad"], "build-clips": ["--out"],
           "rank": ["--out"], "prepare": ["--data-dir"], "import-vault": ["--out"], "train": ["--out"]}


class RecipeError(ValueError):
    pass


def load_recipe(path) -> dict:
    r = json.load(open(path, encoding="utf-8"))
    if not isinstance(r.get("steps"), list) or "adapter" not in r:
        raise RecipeError(f"{path}: a recipe needs 'steps' (a list) and 'adapter'")
    for i, step in enumerate(r["steps"], 1):
        if not step or not all(isinstance(a, str) for a in step):
            raise RecipeError(f"step {i}: a list of strings, starting with a voiceclone subcommand")
        if step[0] not in OUTPUTS:
            raise RecipeError(f"step {i}: '{step[0]}' is not a recipe step (allowed: {', '.join(sorted(OUTPUTS))})")
        if any(a.startswith("--") and "=" in a for a in step):
            raise RecipeError(f"step {i}: write '--flag value' as two strings, not '--flag=value'")
        given = dict(zip(step, step[1:]))
        for flag in OUTPUTS[step[0]]:
            if not given.get(flag, "").startswith("{work}") or ".." in given[flag]:
                raise RecipeError(f"step {i} ({step[0]}): {flag} must be given and be under {{work}}; "
                                  "{data} is read-only")
    if r["adapter"] is not None and not str(r["adapter"]).startswith("{work}"):
        raise RecipeError("'adapter' must be a folder under {work} (or null for zero-shot)")
    return r


def recipe_hash(r: dict) -> str:
    return hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def expand(arg: str, data, work) -> str:
    return arg.replace("{data}", str(data)).replace("{work}", str(work))


def run_recipe(recipe_path, data, work, spec, python: str = sys.executable, run=subprocess.run, log=print) -> Path:
    """Run every step, then benchmark the adapter into WORK/bench (scores.json)."""
    r = load_recipe(recipe_path)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "recipe.json", {**r, "hash": recipe_hash(r)})
    for i, step in enumerate(r["steps"], 1):
        argv = [expand(a, data, work) for a in step]
        log(f"[recipe {i}/{len(r['steps'])}] voiceclone {' '.join(argv)}")
        run([python, "-m", "voiceclone.cli", *argv], check=True)
    bench = ["benchmark", "--spec", str(spec), "--out", str(work / "bench")]
    if r["adapter"] is not None:
        bench += ["--lora", expand(r["adapter"], data, work)]
    log(f"[recipe] voiceclone {' '.join(bench)}")
    run([python, "-m", "voiceclone.cli", *bench], check=True)
    return work / "bench"
