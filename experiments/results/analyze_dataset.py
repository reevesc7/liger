from dataclasses import dataclass
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.figure import Figure
from liger import plotting as pl
from liger import dataset as ds


@dataclass(slots=True)
class Config:
    output_dir: Path
    dataset: str
    dataset_file: Path
    softmax_temperature: float
    agreement_intervals: list[int]


def init_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file path",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser) -> Path:
    args = parser.parse_args()
    return Path(args.config)


def read_config(config_file: str | Path) -> Config:
    with open(config_file, "r") as file:
        cfg = json.load(file)
    dataset_file=Path(cfg["dataset_file"])
    return Config(
        output_dir=Path(cfg["output_dir"]),
        dataset=dataset_file.stem,
        dataset_file=dataset_file,
        softmax_temperature=cfg["softmax_temperature"],
        agreement_intervals=cfg["agreement_intervals"],
    )


def training_variances_fig(
    means: ArrayLike,
    variances: ArrayLike,
    dataset: str,
) -> Figure:
    """Plot the variances of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=np.asarray(variances),
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
    """Plot the expected squared distances to the mode of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=pd.concat((modes, probs), axis=1).apply(_sq_mode_dist, axis=1),
        title=f"{dataset}: ChatGPT responses, expected squared distance to the mode by mean",
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
        x=means,
        y=modes,
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


def _agreements(probs: pd.DataFrame, targets: pd.Series, interval: int = 1) -> pd.Series:
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
        x=means,
        y=_agreements(probs, rounded_means, interval),
        title=f"{dataset}: ChatGPT responses, interval {interval} agreement with mean by mean",
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
        x=variance_stats.index,
        y=variance_stats["mean"],
        error=variance_stats["sem"],
        title=f"{dataset}: ChatGPT responses, variance by mode",
        axis_labels=("ChatGPT mode", "mean of variances (SEM)"),
    )


def training_sq_mode_dists_mode_fig(
    modes: pd.Series,
    probs: pd.DataFrame,
    dataset: str,
) -> Figure:
    mode_dists = pd.Series(pd.concat((modes, probs), axis=1).apply(_sq_mode_dist, axis=1))
    mode_dist_stats = _mean_sem_of_groups(mode_dists, modes)
    return pl.bar(
        x=mode_dist_stats.index,
        y=mode_dist_stats["mean"],
        error=mode_dist_stats["sem"],
        title=f"{dataset}: ChatGPT responses, expected squared distance to mode by mode",
        axis_labels=("ChatGPT mode", "mean of expected squared distance to mode (SEM)"),
    )


def training_means_mode_fig(
    modes: ArrayLike,
    means: pd.Series,
    dataset: str,
) -> Figure:
    mean_stats = _mean_sem_of_groups(means, modes)
    return pl.bar(
        x=mean_stats.index,
        y=mean_stats["mean"],
        error=mean_stats["sem"],
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
        x=agreement_stats.index,
        y=agreement_stats["mean"],
        error=agreement_stats["sem"],
        title=f"{dataset}: ChatGPT responses, interval {interval} agreement with mode by mode",
        axis_labels=("ChatGPT mode", "mean of means (SEM)"),
    )


def training_n_mode_fig(
    modes: pd.Series,
    dataset: str,
) -> Figure:
    ns = modes.groupby(modes).count()
    return pl.bar(
        x=ns.index,
        y=ns,
        title=f"{dataset}: ChatGPT responses, number of responses by mode",
        axis_labels=("ChatGPT mode", "n"),
    )


def make_plots(cfg: Config):
    dataset = ds.Dataset.from_csv(
        cfg.dataset_file,
        "no_match!@#",
        "logprob",
        y_transformers=[
            "liger.probabilities.apply_softmax",
            "liger.probabilities.apply_logprobs_mode",
            "liger.probabilities.apply_logprobs_mean",
            "liger.probabilities.apply_logprobs_variance",
            "liger.probabilities.apply_logprobs_std_dev",
        ],
        y_transformers_kwargs=[
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
        ]
    )
    training_variances_fig(
        dataset.y["mean"],
        dataset.y["variance"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "00_training_variances")
    training_sq_mode_dists_fig(
        dataset.y["mean"],
        dataset.y.loc[:, "mode"],
        dataset.y.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.output_dir / "01_training_sq_mode_dists")
    training_mode_fig(
        dataset.y["mean"],
        dataset.y["mode"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "02_training_modes")
    # training_confidence_fig(
    #     dataset.y["mean"],
    #     dataset.y["std_dev"],
    #     cfg.dataset,
    # ).savefig(cfg.output_dir / "02_training_confidences")
    for interval in cfg.agreement_intervals:
        training_mean_agreement_fig(
            pd.Series(dataset.y["mean"]),
            dataset.y.filter(like="prob"),
            interval,
            cfg.dataset,
        ).savefig(cfg.output_dir / f"03_training_mean_agreements_{interval}")
    training_variances_mode_fig(
        dataset.y["mode"],
        dataset.y.loc[:, "variance"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "10_training_variances_mode")
    training_sq_mode_dists_mode_fig(
        dataset.y.loc[:, "mode"],
        dataset.y.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.output_dir / "11_training_sq_mode_dists_mode")
    training_means_mode_fig(
        dataset.y["mode"],
        dataset.y.loc[:, "mean"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "12_training_means_mode")
    for interval in cfg.agreement_intervals:
        training_mode_agreement_mode_fig(
            dataset.y.loc[:, "mode"],
            dataset.y.filter(like="prob"),
            interval,
            cfg.dataset,
        ).savefig(cfg.output_dir / f"13_training_mode_agreements_{interval}")
    training_n_mode_fig(
        dataset.y.loc[:, "mode"],
        cfg.dataset,
    ).savefig(cfg.output_dir / "14_training_mode_ns")


def main():
    arparser = init_argparser()
    cfg_file = parse_args(arparser)
    cfg = read_config(cfg_file)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    make_plots(cfg)


if __name__ == "__main__":
    main()

