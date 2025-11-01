import torch

def test_as_strided_basic_2x2_view():
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    y = torch.as_strided(x, size=(2, 2), stride=(3, 1), storage_offset=0)
    assert torch.equal(y, x[:2, :2])

def test_as_strided_overlapping_1d_windows():
    # 1D tensor -> overlapping windows of length 3 with step 1
    x = torch.arange(10, dtype=torch.int64)
    # for a contiguous 1D tensor, element stride is 1
    # shape (8,3) with strides (1,1) creates 8 sliding windows of length 3
    y = torch.as_strided(x, size=(8, 3), stride=(1, 1), storage_offset=0)
    # first and last windows should match straightforward slices
    assert torch.equal(y[0], x[0:3])
    assert torch.equal(y[-1], x[7:10])
