import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ..embedding import BaseEmbedder
from ...dataset import Dataset


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

