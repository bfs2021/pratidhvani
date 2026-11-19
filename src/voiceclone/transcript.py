"""Parse a timed transcript export into utterances.

Lines look like `[SS:MM:00] [Speaker] : text`. The timestamp is
seconds:minutes (an export quirk), so `[46:31:00]` means 31 min 46 s.
"""
import re

LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*\[([^\]]+)\]\s*:\s*(.*)$")


def parse_timed_transcript(text: str) -> tuple[list[dict], list[str]]:
    """Returns (utterances [{idx, start, speaker, text}], unparsed lines)."""
    utts, unparsed = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE.match(line)
        if not m:
            unparsed.append(line)
            continue
        ss, mm, _ = (int(x) for x in m.group(1, 2, 3))
        utts.append({"idx": len(utts), "start": mm * 60 + ss,
                     "speaker": m.group(4).strip(), "text": m.group(5).strip()})
    return utts, unparsed


def non_monotonic(utts: list[dict]) -> list[int]:
    """Indices whose start time goes backwards: usually a timestamp typo."""
    return [u["idx"] for a, u in zip(utts, utts[1:]) if u["start"] < a["start"]]
