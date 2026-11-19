"""The hygiene check is what keeps recordings, weights and machine-specific
paths out of the repository without a reviewer having to spot them."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_hygiene", ROOT / "ci" / "check_hygiene.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def check(tmp_path, files: dict):
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return hygiene.problems(tmp_path, list(files))


def test_audio_and_weights_are_refused(tmp_path):
    found = check(tmp_path, {"data/ref.wav": b"RIFF", "models/adapter_model.safetensors": b"x", "a.MP3": b"x"})
    assert len(found) == 3


def test_large_files_are_refused(tmp_path):
    assert check(tmp_path, {"dump.json": b"0" * 2_000_000})


def test_home_paths_and_sys_path_edits_are_refused(tmp_path):
    users, home = "/" + "Users/someone/", "/" + "home/someone/"   # split so this file passes its own check
    found = check(tmp_path, {
        "x.py": f'BASE = "{users}Documents/Projects/x"\nimport sys\nsys.path.insert(0, "..")\n',
        "NOTES.md": f"see [file](file://{home}run.py)\n"})
    assert len(found) == 3


def test_ordinary_code_passes(tmp_path):
    assert check(tmp_path, {"x.py": 'out = Path.home() / "models"  # ~/models\n', "README.md": "use `~/data`\n",
                            "ci.md": "VOICECLONE_DATA=/" + "Users/Shared/voiceclone/data\n"}) == []


def test_the_repository_itself_is_clean():
    import subprocess

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    assert hygiene.problems(ROOT, files) == []
