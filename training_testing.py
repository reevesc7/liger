import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from .dataset import Dataset


# Takes a model, data in Dataset format, indices of data to train on. Returns trained model.
def train_model(model, data: Dataset, train_indices: np.ndarray):
    model.fit(data.X[train_indices], data.y[train_indices])
    return model


# Takes a model, data in Dataset format, indices of data to tset on. Returns performance measures
def evaluate_model(model, data: Dataset, test_indices: np.ndarray) -> tuple[float, float]:
    model_predictions = model.predict(data.X[test_indices])
    mse = mean_squared_error(model_predictions, data.y[test_indices])
    r2 = r2_score(model_predictions, data.y[test_indices])
    return mse, r2


# Returns a model's predictions across all training instances of a KFold cross validation
def kfold_predict(model, kfold: KFold, data: Dataset) -> list[float]:
    predicted = np.zeros(data.y.shape)
    for train_indices, test_indices in kfold.split(data.X):
        model.fit(data.X[train_indices], data.y[train_indices])
        results = model.predict(data.X[test_indices])
        for result_index, dataset_index in enumerate(test_indices):
            predicted[dataset_index] = results[result_index]
    return predicted
