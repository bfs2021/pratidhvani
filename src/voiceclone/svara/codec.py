"""SNAC 24 kHz neural codec (MIT) wrapped for Svara audio tokens."""
import numpy as np

from voiceclone.svara.tokens import frames_to_tokens, tokens_to_frames

SNAC_ID = "hubertsiuzdak/snac_24khz"


def _no_fusion():
    """SNAC's `snake` activation is TorchScript. Once a process has done enough
    other work (seen after LoRA training in the studio), TorchScript tries to
    fuse it, and fusion is not implemented on the Apple GPU ("Unknown device for
    graph fuser"). Running it unoptimised costs nothing that matters here."""
    import torch

    return torch.jit.optimized_execution(False)


class Codec:
    def __init__(self, device: str, model_id: str = SNAC_ID):
        from snac import SNAC

        self.device = device
        self.model = SNAC.from_pretrained(model_id).eval().to(device)

    def encode(self, wav24: np.ndarray) -> list[int]:
        """24 kHz mono float waveform -> vocabulary-offset audio tokens."""
        import torch

        x = torch.from_numpy(wav24).float().view(1, 1, -1).to(self.device)
        with torch.inference_mode(), _no_fusion():
            c0, c1, c2 = [c[0].cpu().tolist() for c in self.model.encode(x)]
        return frames_to_tokens(c0, c1, c2)

    def decode(self, tokens: list[int]) -> np.ndarray:
        """Audio tokens -> 24 kHz float waveform (empty if no whole frame)."""
        import torch

        c0, c1, c2 = tokens_to_frames(tokens)
        if not c0:
            return np.zeros(0, dtype=np.float32)
        codes = [torch.tensor(c, dtype=torch.long, device=self.device).view(1, -1) for c in (c0, c1, c2)]
        with torch.inference_mode(), _no_fusion():
            audio = self.model.decode(codes)
        return audio.view(-1).float().cpu().numpy()
