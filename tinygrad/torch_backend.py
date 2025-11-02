# tinygrad/torch_backend.py
from typing import Tuple
import torch


class TorchBackend:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(self, data):
        """Wrap raw data into a torch tensor."""
        return torch.tensor(data, device=self.device)

    def as_strided(self, input_tensor: torch.Tensor, size, stride):
        """Accurate stride view logic to replace .contiguous() hacks."""
        return torch.as_strided(input_tensor, size=size, stride=stride)

    def contiguous(self, input_tensor: torch.Tensor):
        """Use stride-safe behavior instead of forcing a clone()."""
        if input_tensor.is_contiguous():
            return input_tensor
        shape = tuple(input_tensor.shape)
        strides = tuple(input_tensor.stride())
        return torch.as_strided(input_tensor, size=shape, stride=strides)

    def to_numpy(self, input_tensor: torch.Tensor):
        """Convert back to numpy for interop with Tinygrad core."""
        return input_tensor.cpu().detach().numpy()


def as_view(base: torch.Tensor,
           shape: Tuple[int, ...],
           strides_elems: Tuple[int, ...]) -> torch.Tensor:
    """Return a non-contiguous view over `base` using element strides."""
    return torch.as_strided(base, size=shape, stride=strides_elems)

