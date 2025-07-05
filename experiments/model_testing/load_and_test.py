from statistics import mean
import dill
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from liger.dataset import Dataset
import liger.training_testing as tt


# Dataset
DATASET_PATH = "../main/data/smallville_846.csv"
FEATURE_KEYS = [
    "text-embedding-3-small",
]
SCORE_KEYS = [
    "mean",
    "median",
]

#Training/Testing
N_SPLITS = 5
SCORER_NAMES = [
    "neg_mean_squared_error",
    "tpot.objectives.complexity_scorer",
]
# ^ all scorer names: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers

# This script loads a `fitted_pipeline.pkl`, output from TPOT.
# Change to however you want to introduce your model.
MODEL_PATH = "../main/outputs/smallville_846/2025-07-02_17-22-56.665343/fitted_pipeline.pkl"


def load_model(filepath: str) -> Pipeline:
    """Replace this logic with however you want to initialize your model.
    """
    with open(filepath, "rb") as file:
        return dill.load(file)


def liger_kfold(model: Pipeline, kfold: KFold, scorers: list, dataset: Dataset) -> None:
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
    model = load_model(MODEL_PATH)
    dataset = Dataset.from_csv(DATASET_PATH, FEATURE_KEYS, SCORE_KEYS)
    scorers = [get_scorer(name) for name in tt.init_scorers(SCORER_NAMES)]
    kfold = KFold(N_SPLITS)
    liger_kfold(model, kfold, scorers, dataset)
    #normal_kfold(model, kfold, dataset)


if __name__ == "__main__":
    main()

