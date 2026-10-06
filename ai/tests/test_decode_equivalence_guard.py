import pytest
import torch
from scripts.check_stream_decode import pixel_error


def test_finite_error_and_shape_validation():
    assert pixel_error(torch.zeros(2), torch.zeros(2)) == 0
    for reference, current in [(torch.tensor([float("nan")]), torch.zeros(1)), (torch.zeros(1), torch.tensor([float("inf")])), (torch.zeros(1), torch.zeros(2))]:
        with pytest.raises(ValueError):
            pixel_error(reference, current)
