# torch_backend.py
# WIP: Torch backend for stride-accurate tensor handling
# Part of bounty PR #13041

import torch

class TorchBackend:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(self, data):
        """Wrap raw data into a torch tensor."""
        return torch.tensor(data, device=self.device)

    def as_strided(self, input_tensor, size, stride):
        """Accurate stride view logic to replace .contiguous() hacks."""
        return torch.as_strided(input_tensor, size, stride)

    def contiguous(self, input_tensor):
        """Gracefully handle contiguous behavior."""
        if input_tensor.is_contiguous():
            return input_tensor
        return input_tensor.clone().detach()

    def to_numpy(self, input_tensor):
        """Convert back to numpy for interop with Tinygrad core."""
        return input_tensor.cpu().detach().numpy()
# --- Non-class helper for stride-accurate view creation ---

def as_view(base: torch.Tensor, shape: tuple[int, ...], strides_elems: tuple[int, ...]):
    """
    Return a non-contiguous view over `base` using element strides.
    This avoids unnecessary .contiguous() calls and matches Tinygrad's element-based stride logic.
    """
    return torch.as_strided(base, size=shape, stride=strides_elems)
