import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from .operations import vector_projection, vector_rejection


class Dataset:
    def __init__(self, n_entries: int, X_len: int, y_len: int = 1):
        self.X = np.zeros((n_entries, X_len))
        if y_len == 1:
            self.y = np.zeros(n_entries)
        else:
            self.y = np.zeros((n_entries, y_len))


    @classmethod
    def from_df(self, df: pd.DataFrame) -> 'Dataset':
        X_len = len(df['X'][0].strip('[]').split(", "))
        y_len = len(df['y'][0].strip('[]').split(", "))
        dataset = Dataset(df.shape[0], X_len, y_len)
        for i in range(len(df['y'])):
            dataset.X[i] = [float(x) for x in df['X'][i].strip('[]').split(", ")]
            dataset.y[i] = [float(y) for y in df['y'][i].strip('[]').split(", ")]
        return dataset


    @classmethod
    def from_csv(self, filename: str) -> 'Dataset':
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


    def analyze_manifold(self, pointA: ArrayLike, pointB: ArrayLike) -> pd.DataFrame:
        manifold = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ['y', 'alpha', 'dist']})
        ABvector = pointB - pointA
        manifold['y'] = self.y
        for i, pointP in enumerate(self.X):
            APvector = pointP - pointA
            manifold['alpha'][i] = (vector_projection(ABvector, APvector)*8)+1
            manifold['dist'][i] = vector_rejection(ABvector, APvector)
        return manifold