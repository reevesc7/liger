import pandas as pd
import re
from .. import Dataset


# Replaces a string in every entry of the data with a different, given string.
def change_sentences(data: pd.DataFrame, str_to_change: str, new_str: str) -> pd.DataFrame:
    return pd.DataFrame({column: [re.sub(str_to_change, new_str, entry) if type(entry)==str else entry for entry in data.loc[:, column]] for column in data.columns})


# Adds a string to the beginning of every entry.
def add_sentences(data: pd.DataFrame, str_to_add: str) -> pd.DataFrame:
    return pd.DataFrame({column: [str_to_add.strip() + " " + entry if type(entry)==str else entry for entry in data.loc[:, column]] for column in data.columns})


class BaseEmbedder:
    def set_model(self, model_str: str) -> None:
        self.model = model_str


    def embed_dataframe(self, data: pd.DataFrame, feature_keys: pd.Index, score_key: str) -> Dataset:
        raise NotImplementedError()
