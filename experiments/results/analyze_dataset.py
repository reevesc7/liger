import sys
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.figure import Figure
from liger import plotting as pl
from liger import dataset as ds
from liger import transforms as ts


@dataclass(slots=True)
class Config:
    dataset: str
    output_dir: Path
    dataset_file: Path
    softmax_temperature: float
    agreement_intervals: list[int]

    def __init__(
        self,
        cfg_path: Path,
    ) -> None:
        self.dataset = cfg_path.stem
        self.output_dir = cfg_path.parent / self.dataset
        parsed = tomllib.load(cfg_path.open("rb"))
        self.dataset_file = Path(parsed["dataset_file"])
        self.softmax_temperature = parsed["softmax_temperature"]
        self.agreement_intervals = parsed["agreement_intervals"]


def load_dataset(cfg: Config) -> pd.DataFrame:
    columns = r"^logprob"
    strip = "response_logprob_"
    temp = cfg.softmax_temperature
    return ds.multi_data_from_csv(cfg.dataset_file, (
        {
            "columns": columns,
            "transformer": lambda d: ts.apply_softmax(d, temp, strip),
        },
        {
            "columns": columns,
            "transformer": lambda d: ts.apply_logprobs_mode(d, temp, strip),
        },
        {
            "columns": columns,
            "transformer": lambda d: ts.apply_logprobs_mean(d, temp, strip),
        },
        {
            "columns": columns,
            "transformer": lambda d: ts.apply_logprobs_variance(d, temp, strip),
        },
        {
            "columns": columns,
            "transformer": lambda d: ts.apply_logprobs_std_dev(d, temp, strip),
        },
    ))


def training_variances_fig(
    means: ArrayLike,
    variances: ArrayLike,
    dataset: str,
) -> Figure:
    """Plot the variances of LLM response distributions, across means.
    """
    return pl.scatter(
        data=[np.array((means, variances))],
        title=f"{dataset}: ChatGPT responses, variance by mean",
        axis_labels=("mean", "variance"),
        trend_orders=[],
        plot_perfect=False,
    )


def _sq_mode_dist(row: pd.Series) -> float:
    prob_vals = [
        int(index[index.find("_") + 1:])
        for index in row.index.array
        if "prob_" in index
    ]
    return sum(
        float(row[f"prob_{i}"]) * (i - float(row["mode"])) ** 2
        for i in prob_vals
    )


def training_sq_mode_dists_fig(
    means: ArrayLike,
    modes: pd.Series,
    probs: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot the expected squared distances to the mode of LLM response distributions,
    across means.
    """
    return pl.scatter(
        data=[np.array((
            means,
            pd.concat((modes, probs), axis=1).apply(_sq_mode_dist, axis=1),
        ))],
        title=f"{dataset}: ChatGPT responses, "
            "expected squared distance to the mode by mean",
        axis_labels=("mean", "sq_mode_dist"),
        trend_orders=[],
        plot_perfect=False,
    )


def training_mode_fig(
    means: ArrayLike,
    modes: ArrayLike,
    dataset: str,
) -> Figure:
    """Plot the modes of LLM response distributions, across means.
    """
    return pl.scatter(
        data=[np.array((means, modes))],
        title=f"{dataset}: ChatGPT responses, mode by mean",
        axis_labels=("mean", "mode"),
        trend_orders=[],
        plot_perfect=False,
    )


# def training_confidence_fig(
#     means: ArrayLike,
#     std_devs: ArrayLike,
#     dataset: str,
# ) -> Figure:
#     """Plot the confidences of LLM response distributions, across means.
#     """
#     return pl.scatter(
#         x=means,
#         y=np.full(std_devs.shape, 1) - np.asarray(std_devs) ** 2 / 20.25,
#         title=f"{dataset}: ChatGPT responses, confidence by mean",
#         axis_labels=("mean", "confidence"),
#         trend_orders=[],
#         plot_perfect=False,
#     )


def _agreement(row: pd.Series, interval: int = 1) -> float:
    response = int(row.iloc[0])
    agreements: list[float] = []
    for i in range(max(response - interval, 1), min(response + interval + 1, 11)):
        agreement = row.get(f"prob_{i}", 0.0)
        assert agreement is not None, "Series must have been empty... somehow"
        agreements.append(float(agreement))
    return sum(agreements)


def _agreements(
    probs: pd.DataFrame,
    targets: pd.Series,
    interval: int = 1,
) -> pd.Series:
    data = pd.concat((targets, probs), axis=1)
    return pd.Series(data.apply(lambda row: _agreement(row, interval), axis=1))


def training_mean_agreement_fig(
    means: pd.Series,
    probs: pd.DataFrame,
    interval: int,
    dataset: str,
) -> Figure:
    """Plot the agreements of LLM response distributions, across means.
    """
    rounded_means = pd.Series(means.apply(lambda row: round(row)))
    return pl.scatter(
        data=[np.array((means, _agreements(probs, rounded_means, interval)))],
        title=f"{dataset}: ChatGPT responses, interval "
            f"{interval} agreement with mean by mean",
        axis_labels=("mean", "agreement"),
        trend_orders=[],
        plot_perfect=False,
    )


def _sem(row: pd.Series) -> pd.Series:
    return pd.Series(
        (row["mean"], row["std"] / row["count"] ** 0.5),
        index=["mean", "sem"],
    )


def _mean_sem_of_groups(samples: pd.Series, groups: ArrayLike) -> pd.DataFrame:
    """Compute the mean and standard error of the mean for grouped samples.
    """
    return pd.DataFrame(samples.groupby(groups).agg([
        "mean",
        "std",
        "count",
    ]).apply(_sem, axis=1).fillna(0.0))


def training_variances_mode_fig(
    modes: ArrayLike,
    variances: pd.Series,
    dataset: str,
) -> Figure:
    variance_stats = _mean_sem_of_groups(variances, modes)
    return pl.bar(
        data=[np.array((
            variance_stats.index,
            variance_stats["mean"],
            variance_stats["sem"],
        ))],
        title=f"{dataset}: ChatGPT responses, variance by mode",
        axis_labels=("ChatGPT mode", "mean of variances (SEM)"),
    )


def training_sq_mode_dists_mode_fig(
    modes: pd.Series,
    probs: pd.DataFrame,
    dataset: str,
) -> Figure:
    mode_dists = pd.Series(
        pd.concat((modes, probs), axis=1).apply(_sq_mode_dist, axis=1)
    )
    mode_dist_stats = _mean_sem_of_groups(mode_dists, modes)
    return pl.bar(
        data=[np.array((
            mode_dist_stats.index,
            mode_dist_stats["mean"],
            mode_dist_stats["sem"],
        ))],
        title=f"{dataset}: ChatGPT responses, "
            "expected squared distance to mode by mode",
        axis_labels=("ChatGPT mode", "mean of expected squared distance to mode (SEM)"),
    )


def training_means_mode_fig(
    modes: ArrayLike,
    means: pd.Series,
    dataset: str,
) -> Figure:
    mean_stats = _mean_sem_of_groups(means, modes)
    return pl.bar(
        data=[np.array((
            mean_stats.index,
            mean_stats["mean"],
            mean_stats["sem"],
        ))],
        title=f"{dataset}: ChatGPT responses, mean by mode",
        axis_labels=("ChatGPT mode", "mean of means (SEM)"),
    )


def training_mode_agreement_mode_fig(
    modes: pd.Series,
    probs: pd.DataFrame,
    interval: int,
    dataset: str,
) -> Figure:
    agreements = _agreements(probs, modes, interval)
    agreement_stats = _mean_sem_of_groups(agreements, modes)
    return pl.bar(
        data=[np.array((
            agreement_stats.index,
            agreement_stats["mean"],
            agreement_stats["sem"],
        ))],
        title=f"{dataset}: ChatGPT responses, "
            f"interval {interval} agreement with mode by mode",
        axis_labels=("ChatGPT mode", "mean of means (SEM)"),
    )


def training_n_mode_fig(
    modes: pd.Series,
    dataset: str,
) -> Figure:
    ns = modes.groupby(modes).count()
    return pl.bar(
        data=[np.array((ns.index, ns))],
        title=f"{dataset}: ChatGPT responses, number of responses by mode",
        axis_labels=("ChatGPT mode", "n"),
    )


def make_plots(cfg: Config, dataset: pd.DataFrame):
    training_variances_fig(
        dataset["mean"],
        dataset["variance"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "00_training_variances")
    training_sq_mode_dists_fig(
        dataset["mean"],
        dataset.loc[:, "mode"],
        dataset.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.output_dir / "01_training_sq_mode_dists")
    training_mode_fig(
        dataset["mean"],
        dataset["mode"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "02_training_modes")
    # training_confidence_fig(
    #     dataset["mean"],
    #     dataset["std_dev"],
    #     cfg.dataset,
    # ).savefig(cfg.output_dir / "02_training_confidences")
    for interval in cfg.agreement_intervals:
        training_mean_agreement_fig(
            pd.Series(dataset["mean"]),
            dataset.filter(like="prob"),
            interval,
            cfg.dataset,
        ).savefig(cfg.output_dir / f"03_training_mean_agreements_{interval}")
    training_variances_mode_fig(
        dataset["mode"],
        dataset.loc[:, "variance"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "10_training_variances_mode")
    training_sq_mode_dists_mode_fig(
        dataset.loc[:, "mode"],
        dataset.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.output_dir / "11_training_sq_mode_dists_mode")
    training_means_mode_fig(
        dataset["mode"],
        dataset.loc[:, "mean"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "12_training_means_mode")
    for interval in cfg.agreement_intervals:
        training_mode_agreement_mode_fig(
            dataset.loc[:, "mode"],
            dataset.filter(like="prob"),
            interval,
            cfg.dataset,
        ).savefig(cfg.output_dir / f"13_training_mode_agreements_{interval}")
    training_n_mode_fig(
        dataset.loc[:, "mode"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "14_training_mode_ns")


def main():
    cfg = Config(Path(sys.argv[1]))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(cfg)
    make_plots(cfg, dataset)


if __name__ == "__main__":
    main()
