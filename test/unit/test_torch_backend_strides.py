import torch


def as_view(base: torch.Tensor, shape, strides):
    """
    Tiny helper to mimic a stride-accurate view (what we're trying to do in the PR).
    We keep it in the test so we can prove the idea without touching tinygrad core yet.
    """
    return torch.as_strided(base, size=shape, stride=strides)


def test_as_view_simple_row_major():
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    # row-major view, should match base
    view = as_view(base, (2, 3), base.stride())
    assert view.is_contiguous()
    assert torch.equal(view, base)


def test_as_view_slice_like():
    base = torch.arange(10, dtype=torch.float32)
    # take 3 elements, stride 2 -> [0, 2, 4]
    view = as_view(base, (3,), (2,))
    expected = torch.tensor([0., 2., 4.])
    assert not view.is_contiguous()
    assert torch.equal(view, expected)


def test_as_view_transpose():
    # this was the one that failed before
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # transpose should be shape (4, 3) and strides (1, 4)
    view = as_view(base, (4, 3), (1, 4))
    assert not view.is_contiguous()
    assert torch.equal(view, base.t())

