import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from .operations import vector_projection, vector_rejection


class Dataset:
    def __init__(self, n_entries: int, entry_length: int):
        self.X = np.zeros((n_entries,entry_length))
        self.y = np.zeros(n_entries)


    @classmethod
    def from_csv(self, dfname: str) -> 'Dataset':
        df = pd.read_csv(dfname)
        X_columns = [column for column in df.columns if column.isnumeric()]
        dataset = Dataset(df.shape[0], len(X_columns))
        dataset.X = np.array([df[column] for column in X_columns]).transpose()
        dataset.y = df['y']
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