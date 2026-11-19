"""Checks every tracked file against the repository rules in CONTRIBUTING.md
that a machine can check:
  - no audio, model weights or large files (recordings and weights live in
    the shared data folder, never in git)
  - no paths into someone's home folder (everything is a CLI argument)
  - no sys.path edits (use the installed package)
  - ci/recipe.json is a valid recipe
Usage: python ci/check_hygiene.py   (from the repository root; exit 1 on problems)
"""
import re
import subprocess
import sys
from pathlib import Path

MEDIA = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4",
         ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".npz", ".npy", ".pkl"}
MAX_BYTES = 1_000_000
TEXT = {".py", ".md", ".json", ".jsonl", ".toml", ".yml", ".yaml", ".sh", ".txt", ".cfg", ".ini", ".html", ".js", ".css"}
HOME_PATH = re.compile(r"(?<![\w.])(/Users/(?!Shared/)[A-Za-z][\w.-]*/|/home/[a-z][\w.-]*/|[A-Za-z]:\\Users\\)")
SYS_PATH = re.compile(r"\bsys\.path\.(insert|append|extend)\(|\bsys\.path\s*\+?=")
SELF = Path(__file__).resolve()


def problems(root: Path, files: list[str]) -> list[str]:
    out = []
    for f in files:
        p = root / f
        if not p.is_file():
            continue
        if p.suffix.lower() in MEDIA:
            out.append(f"{f}: audio or model files do not belong in git")
            continue
        if p.stat().st_size > MAX_BYTES:
            out.append(f"{f}: {p.stat().st_size / 1e6:.1f} MB; large files do not belong in git")
        if p.suffix.lower() not in TEXT or p.resolve() == SELF:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if HOME_PATH.search(line):
                out.append(f"{f}:{n}: path into a home folder; take it as an argument instead")
            if p.suffix == ".py" and SYS_PATH.search(line):
                out.append(f"{f}:{n}: sys.path edit; install the package (uv pip install -e .) instead")
    recipe = root / "ci" / "recipe.json"
    if recipe.exists():
        from voiceclone.recipe import RecipeError, load_recipe

        try:
            load_recipe(recipe)
        except (RecipeError, ValueError) as e:
            out.append(f"ci/recipe.json: {e}")
    return out


def main():
    root = Path.cwd()
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    found = problems(root, files)
    for line in found:
        print(line)
    print(f"{len(files)} files checked, {len(found)} problems")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
