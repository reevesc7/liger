import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from liger import plotting as pl
from liger import dataset as ds
import experiments.results.config as cfg


def plot_training_variances(means: ArrayLike, std_devs: ArrayLike):
    """Plot the standard deviation of LLM response distributions, across means.
    """
    plot = pl.scatter(
        x=means,
        y=std_devs,
        title=f"{cfg.DATASET}: ChatGPT responses, standard deviation by mean",
        axis_labels=("mean", "std_dev"),
        trend_orders=[],
        plot_perfect=False,
    )
    plot.savefig(cfg.DIRECTORY / "1_training_variances")


def plot_responses(responses: pd.DataFrame, means: pd.DataFrame):
    """Plot predicted means against training means.
    """
    plot = pl.scatter(
        x=np.transpose(means),
        y=np.transpose(responses),
        title=f"{cfg.DATASET}: predicted means against training means",
        axis_labels=("ChatGPT mean", "predicted"),
        trend_orders=[5],
        plot_perfect=True,
    )
    plot.savefig(cfg.DIRECTORY / "2_responses")


def plot_abs_errors(responses: pd.DataFrame, means: pd.DataFrame) -> None:
    """Plot absolute errors of predictions across training means.
    """
    abs_errors = np.absolute(responses - means)
    plot = pl.scatter(
        x=np.transpose(means),
        y=np.transpose(abs_errors),
        title=f"{cfg.DATASET}: absolute errors of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted absolute error"),
        trend_orders=[5],
    )
    plot.savefig(cfg.DIRECTORY / "3_abs_errors")


def plot_squared_errors(responses: pd.DataFrame, means: pd.DataFrame) -> None:
    """Plot squared errors of predictions across training means.
    """
    squared_errors = np.square(responses - means)
    plot = pl.scatter(
        x=np.transpose(means),
        y=np.transpose(squared_errors),
        title=f"{cfg.DATASET}: squared errors of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted squared error"),
        trend_orders=[5],
    )
    plot.savefig(cfg.DIRECTORY / "4_squared_errors")


def plot_zscores(responses: pd.DataFrame, means: pd.DataFrame, std_devs: pd.DataFrame) -> None:
    """Plot Z-scores of predictions across training means.
    """
    zscores = np.clip((responses - means) / std_devs, -20, 20)
    plot = pl.scatter(
        x=np.transpose(means),
        y=np.transpose(zscores),
        title=f"{cfg.DATASET}: Z-scores of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted Z-score"),
        trend_orders=[5],
    )
    plot.savefig(cfg.DIRECTORY / "5_zscores")


def main():
    dataset = ds.Dataset.from_csv(cfg.DATASET_PATH, "no_match!@#", ["mean", "std_dev"])
    responses = pd.read_csv(cfg.RESPONSES)
    means = pd.concat([pd.Series(dataset.y["mean"])] * responses.shape[1], axis=1)
    std_devs = pd.concat([pd.Series(dataset.y["std_dev"])] * responses.shape[1], axis=1)
    means.columns = responses.columns
    std_devs.columns = responses.columns
    plot_training_variances(dataset.y["mean"], dataset.y["std_dev"])
    plot_responses(responses, means)
    plot_abs_errors(responses, means)
    plot_squared_errors(responses, means)
    plot_zscores(responses, means, std_devs)


if __name__ == "__main__":
    main()
