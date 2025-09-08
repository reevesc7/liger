from pathlib import Path
import pandas as pd
from liger import plotting as pl
from liger import dataset as ds


#DIRECTORY = Path("smallville_846")
#ACTUAL = "../tpot/data/smallville_846.csv"
#Y = [
#    "mean",
#]
#RESPONSES = "responses.csv"
#TITLE = "smallville_846 predicted values"
DIRECTORY = Path("smallville_765")
ACTUAL = "../tpot/data/smallville_765.csv"
Y = [
    "mean",
]
RESPONSES = "responses.csv"
TITLE = "smallville_765 predicted values"


def main():
    """Only works with single-value prediction, for "mean" column
    """
    actual = ds.Dataset.from_csv(ACTUAL, "nothing_#15@", Y)
    responses = pd.read_csv(DIRECTORY / RESPONSES)
    plot = pl.scatter(
        data=[[actual.y["mean"].to_numpy(), responses[column].to_numpy()] for column in responses.columns],
        title=TITLE,
        axis_labels=("ChatGPT mean", "predicted"),
        trend_orders=[5],
        plot_perfect=True,
    )
    plot.savefig(DIRECTORY / "plot")


if __name__ == "__main__":
    main()
