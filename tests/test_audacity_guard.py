import pytest

from voiceclone.enhance import audacity


def test_refuses_to_quit_an_open_audacity(tmp_path, monkeypatch):
    # quitting someone's open Audacity loses their unsaved project
    monkeypatch.setattr(audacity, "is_running", lambda app: True)
    restarted = []
    monkeypatch.setattr(audacity, "_restart", lambda app, pipe: restarted.append(app))
    with pytest.raises(RuntimeError, match="already running"):
        audacity.audacity_denoise(tmp_path / "talk.mp3", tmp_path / "work")
    assert restarted == []                      # never got as far as quitting it
    assert not (tmp_path / "work").exists()     # and did no work first


def test_force_restart_is_allowed_past_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(audacity, "is_running", lambda app: True)

    def stop(*a, **k):
        raise KeyboardInterrupt   # stop at the first real action after the guard

    monkeypatch.setattr("voiceclone.audio.ffmpeg_decode", stop)
    with pytest.raises(KeyboardInterrupt):
        audacity.audacity_denoise(tmp_path / "talk.mp3", tmp_path / "work", force_restart=True)
