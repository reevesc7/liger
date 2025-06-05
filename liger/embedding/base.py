# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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


    def embed_dataframe(
        self,
        data: pd.DataFrame,
        feature_keys: pd.Index,
        score_key: pd.Index
    ) -> Dataset:
        raise NotImplementedError()
