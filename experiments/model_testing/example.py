from statistics import mean
import dill
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from liger.dataset import Dataset
import liger.training_testing as tt


# MODEL: loading/creating a model
MODEL_PATH = "../tpot/outputs/test_data/2025-07-05_00-49-55.061326/fitted_pipeline.pkl"

# DATASET: filepath and column selection to create a dataset
DATASET_PATH = "../tpot/data/test_data.csv"
FEATURE_KEYS = [
    "all-mpnet-base-v2",
]
SCORE_KEYS = [
    "mean",
    "median",
]

#TRAINING/TESTING: cross-validation and scoring functions
N_SPLITS = 5
SCORER_NAMES = [
    "neg_mean_squared_error",
    "tpot.objectives.complexity_scorer",
]
# ^ all scikit-learn scorer names:
#   https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
#   Additionally, providing the path to a function will load the function and attempt
#   to use it as a scorer or objective.
#   Format: "<module>.<submodule>.<subsubmodule>...<function>"


def get_model(model_path: str) -> BaseEstimator:
    """Replace this logic with however you want to initialize your model.
    """
    with open(model_path, "rb") as file:
        return dill.load(file)


def liger_kfold(model: BaseEstimator, kfold: KFold, scorers: list, dataset: Dataset) -> None:
    """The liger KFold process, which gives scores for each scorer fed in.
    """
    _, kfold_scores = tt.kfold_predict(model, kfold, scorers, dataset)
    print("KFold score(s):", {SCORER_NAMES[i]: kfold_scores[i] for i in range(len(SCORER_NAMES))})


def normal_kfold(model, kfold: KFold, dataset: Dataset) -> None:
    """If you want to do normal fitting, this is how you index `Dataset` entries.

    I think `Estimator.score()` returns the R^2 value for the model's predictions.
    """
    scores = []
    for train_indices, test_indices in kfold.split(dataset.x):
        model.fit(dataset.x.iloc[train_indices], dataset.y.iloc[train_indices])
        scores.append(model.score(dataset.x.iloc[test_indices], dataset.y.iloc[test_indices]))
    print("Score:", mean(scores))


def main():
    model = get_model(MODEL_PATH)
    dataset = Dataset.from_csv(DATASET_PATH, FEATURE_KEYS, SCORE_KEYS)
    scorers = [get_scorer(name) for name in tt.init_scorers(SCORER_NAMES)]
    kfold = KFold(N_SPLITS)
    liger_kfold(model, kfold, scorers, dataset)
    #normal_kfold(model, kfold, dataset)


if __name__ == "__main__":
    main()

