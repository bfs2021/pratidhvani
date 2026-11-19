## What and why

<!-- One or two sentences. -->

## Experiment card (delete for code-only changes)

- **Hypothesis:** <!-- what you expect to improve, and why -->
- **The one change:** <!-- data / filter / training setting / denoiser; everything else as on main -->
- **Expected effect on the gate metrics:** <!-- e.g. similarity up, CER unchanged -->
- **Listening test:** <!-- needed for accent claims: listeners, trials, result of `compare --answers` -->

## Checklist

- [ ] `pytest` and `python ci/check_hygiene.py` pass locally
- [ ] No audio, weights, transcripts of the speaker, personal names or local paths in the diff
- [ ] Training transcripts are verbatim (IndicConformer or human), not Whisper
- [ ] Labelled `experiment` if this claims to improve the voice
