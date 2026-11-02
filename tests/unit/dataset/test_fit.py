import pytest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
from liger.dataset import fit


SAMPLE_DATA = [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.5, 1.5, 1.5],
    [3.0, 3.0, 3.0],
    [1.0, 2.0, 1.0],
]
C_1_3 = 1.0 / 3.0
ALPHAS = [0.0, 1.0, 0.5, 2.0, C_1_3]
DEVIATIONS = [0.0, 0.0, 0.0, 0.0, np.linalg.norm((
    0.0 - C_1_3,
    1.0 - C_1_3,
    0.0 - C_1_3,
))]


@pytest.fixture
def sample_data_array() -> np.ndarray:
    return np.array((SAMPLE_DATA, SAMPLE_DATA))


@pytest.fixture
def sample_data_frame() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE_DATA)


def test_linear_fit_array(sample_data_array: np.ndarray) -> None:
    alphas, deviations = fit.linear_fit_array(
        sample_data_array,
        sample_data_array[0][0],
        sample_data_array[0][1],
    )
    assert isinstance(alphas, np.ndarray)
    assert isinstance(deviations, np.ndarray)
    assert alphas.shape == (2, 5)
    assert deviations.shape == (2, 5)
    expected_alphas = np.array((ALPHAS, ALPHAS))
    expected_deviations = np.array((DEVIATIONS, DEVIATIONS))
    assert_array_equal(alphas, expected_alphas)
    assert_array_equal(deviations, expected_deviations)


def test_linear_fit(sample_data_frame: pd.DataFrame) -> None:
    linear_fit = fit.linear_fit(
        sample_data_frame,
        sample_data_frame.loc[0],
        sample_data_frame.loc[1],
    )
    assert isinstance(linear_fit, pd.DataFrame)
    assert linear_fit.shape == (5, 2)
    expected = pd.DataFrame({"alpha": ALPHAS, "deviation": DEVIATIONS})
    assert_frame_equal(linear_fit, expected)
