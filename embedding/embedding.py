import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from ..dataset import Dataset


# Replaces a string in every entry of the data with a different, given string.
def change_sentences(data: pd.DataFrame, str_to_change: str, new_str: str) -> pd.DataFrame:
    return pd.DataFrame({column: [re.sub(str_to_change, new_str, entry) if type(entry)==str else entry for entry in data.loc[:, column]] for column in data.columns})


# Adds a string to the beginning of every entry.
def add_sentences(data: pd.DataFrame, str_to_add: str) -> pd.DataFrame:
    return pd.DataFrame({column: [str_to_add.strip() + " " + entry if type(entry)==str else entry for entry in data.loc[:, column]] for column in data.columns})


# Generates a dataset from the specified rows of a DataFrame.
def embed_dataframe(model: SentenceTransformer, data: pd.DataFrame, feature_keys: pd.Index, score_key: str, model_dim: int = 384) -> Dataset:
    n_entries = data.shape[0]
    n_features = len(feature_keys)
    feature_vectors = np.zeros((n_features,n_entries,model_dim))
    for feature_index, feature_key in enumerate(feature_keys):
        feature_vectors[feature_index] = model.encode(data[feature_key])
    dataset = Dataset(n_entries, n_features*model_dim)
    dataset.X = feature_vectors.transpose((1,0,2)).reshape((n_entries,-1))
    dataset.y = np.array(data[score_key])
    return dataset
