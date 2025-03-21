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
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder
from ..dataset import Dataset


class STEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_str: str,
    ):
        self.model = SentenceTransformer(model_str)


    # Generates a dataset from the specified rows of a DataFrame.
    def embed_dataframe(self, data: pd.DataFrame, feature_keys: pd.Index, score_key: str) -> Dataset:
        n_entries = data.shape[0]
        n_features = len(feature_keys)

        # Hate how hacky this is
        feature_vectors = None
        for feature_index, feature_key in enumerate(feature_keys):
            embedding = np.array(self.model.encode(np.array(data[feature_key])))
            if feature_vectors is None:
                feature_vectors = np.zeros((n_features, n_entries, embedding.shape[1]))
            feature_vectors[feature_index] = embedding
        feature_vectors = np.array(feature_vectors)
        dataset = Dataset(n_entries, n_features*feature_vectors.shape[2])
        dataset.X = feature_vectors.transpose((1, 0, 2)).reshape((n_entries, -1))
        dataset.y = np.array(data[score_key])
        return dataset

