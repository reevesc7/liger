from pathlib import Path
import pandas as pd
from liger import dataset as ds


def _apply_zero(
    data: pd.DataFrame,
) -> pd.Series:
    assert data.shape == (91, 768)
    return pd.Series([0 for _ in range(data.shape[0])], name="zero")


def test_data_from_csv(example_data_file: Path) -> None:
    data = ds.data_from_csv(file_path=example_data_file)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (91, 1)
    assert data.columns[0] == "prompt"
    data = ds.data_from_csv(
        file_path=example_data_file,
        columns=r"^all-mpnet-base-v2_\d*$",
        transformer=_apply_zero,
    )
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (91, 1)
    data = ds.data_from_csv(
        file_path=example_data_file,
        columns=("mean", "prompt"),
    )
    assert data.shape == (91, 2)
    assert data.columns[0] == "prompt"
    data = ds.data_from_csv(
        file_path=example_data_file,
        columns=[55, 0, 1],
    )
    assert data.shape == (91, 3)
    assert data.columns[0] == "prompt"
    assert data.columns[2] == "all-mpnet-base-v2_52"
