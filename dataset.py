import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from .operations import vector_projection, vector_rejection


class Dataset:
    def __init__(self, n_entries: int, X_len: int, y_len: int = 1):
        self.X = np.array(np.zeros((n_entries, X_len)))
        if y_len == 1:
            self.y = np.array(np.zeros(n_entries))
        else:
            self.y = np.array(np.zeros((n_entries, y_len)))


    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Dataset':
        X_len = len(str(df['X'][0]).strip('[]').split(", "))
        y_len = len(str(df['y'][0]).strip('[]').split(", "))
        dataset = Dataset(df.shape[0], X_len, y_len)
        dataset.X = np.array(df['X'].apply(_format_df_list))
        if y_len > 1:
            dataset.y = np.array(df['y'].apply(_format_df_list))
        else:
            dataset.y = np.array(df['y'])
        # for i in range(len(df['y'])):
        #     dataset.X[i] = np.array([float(x) for x in str(df['X'][i]).strip('[]').split(", ")])
        #     if y_len > 1:
        #         dataset.y[i] = np.array([float(y) for y in str(df['y'][i]).strip('[]').split(", ")])
        # if y_len == 1:
        #     dataset.y = np.array(df['y'])
        return dataset


    @classmethod
    def from_csv(cls, filename: str) -> 'Dataset':
        df = pd.read_csv(filename)
        dataset = Dataset.from_df(df)
        return dataset
    

    def __str__(self):
        return f"X: {self.X},\ny: {self.y}"


    def flatten(self) -> pd.DataFrame:
        df = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in (['y'] + [i for i in range(self.X.shape[1])])})
        df['y'] = self.y
        for i, dimension in enumerate(self.X.transpose()):
            df[i] = dimension
        return df


    def to_csv(self, filename):
        self.flatten().to_csv(filename, index=False)


    def analyze_manifold(self, point_a: ArrayLike, point_b: ArrayLike) -> pd.DataFrame:
        manifold = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ['y', 'alpha', 'dist']})
        ab_vector = np.subtract(point_b, point_a)
        manifold['y'] = self.y
        for i, point_p in enumerate(self.X):
            ap_vector = np.subtract(point_p, point_a)
            manifold['alpha'][i] = (vector_projection(ab_vector, ap_vector)*8)+1
            manifold['dist'][i] = vector_rejection(ab_vector, ap_vector)
        return manifold


def _format_df_list(entry: str) -> np.ndarray:
    split_entry = entry.strip("[ ]").split(", ")
    try:
        return np.array(split_entry, dtype=np.int64)
    except ValueError:
        return np.array(split_entry, dtype=np.float64)

