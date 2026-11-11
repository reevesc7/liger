import pytest
import numpy as np
from numpy.testing import assert_array_equal
from liger import pareto


SAMPLE_DATA = [
    [1.0, 3.0, -3.0],
    [0.0, 2.0, -2.0],
    [1.5, 1.5, -1.5],
    [3.0, 1.0, -4.0],
    [1.0, 0.5, -0.5],
    [0.0, 0.0, -2.0],
]
PARETO_MASK = [True, True, True, True, True, False]
KNEE_INDEX = 2


@pytest.fixture
def sample_data() -> list[list[float]]:
    return SAMPLE_DATA


def test_pareto_mask(sample_data: list[list[float]]) -> None:
    mask = pareto.pareto_mask(sample_data)
    assert isinstance(mask, np.ndarray)
    assert_array_equal(mask, np.array(PARETO_MASK))


def test_pareto_knee_index(sample_data: list[list[float]]) -> None:
    knee_index = pareto.pareto_knee_index(sample_data)
    assert isinstance(knee_index, int)
    assert knee_index == KNEE_INDEX
