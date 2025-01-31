from sys import platform
if platform == 'win32':
    from os import environ
    environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from . import pipelines as pl
from argparse import ArgumentParser

def main():
    args = get_args()

    training_data = args.data
    output_directory = args.output
    regression = args.regression
    n_generations = args.ngens
    pop_size = args.popsize
    tpot_random_states = [i+args.stpotrs for i in range(args.ntpotrs)]
    eval_random_states = [i+args.sevalrs for i in range(args.nevalrs)]
    n_split_generations = args.nsplitgens
    no_trees = args.notrees
    no_xg = args.noxg
    cutoff_mins = args.cutoffmins
    n_splits = args.nsplits

    pl.tpot_pipeline(
        training_data,
        output_directory,
        regression,
        n_generations,
        pop_size,
        tpot_random_states,
        eval_random_states,
        n_split_generations,
        no_trees,
        no_xg,
        cutoff_mins,
        n_splits,
    )


def get_args() -> ArgumentParser:
    parser = ArgumentParser(description="Script that runs tpot fitting and evaluation")
    parser.add_argument("--data", type=str, required=True, help="training data file path")      #"Data/human_size_rating_1_1.csv"
    parser.add_argument("--output", type=str, required=False, default="Outputs/tpot/", help="output directory")
    parser.add_argument("--regression", action="store_true", dest="regression", help="run regression (not classification)")
    parser.add_argument("--ngens", type=int, required=False, default=5, help="number of tpot fitting generations")
    parser.add_argument("--popsize", type=int, required=False, default=50, help="tpot fitting population size")
    parser.add_argument("--stpotrs", type=int, required=False, default=0, help="tpot random state start")
    parser.add_argument("--ntpotrs", type=int, required=False, default=1, help="number of tpot random states to test")
    parser.add_argument("--sevalrs", type=int, required=False, default=0, help="eval random state start")
    parser.add_argument("--nevalrs", type=int, required=False, default=1, help="number of eval random states to test")
    parser.add_argument("--nsplitgens", type=int, required=False, default=None, help="number of generations to run before pickling tpot object")
    parser.add_argument("--notrees", action="store_true", dest="notrees", help="whether RandomForest and Tree models should be excluded")
    parser.add_argument("--noxg", action="store_true", dest="noxg", help="whether XGBoost models should be excluded")
    parser.add_argument("--cutoffmins", type=int, required=False, default=None, help="minutes at which tpot should terminate fitting early")
    parser.add_argument("--nsplits", type=int, required=False, default=None, help="number of splits for K-fold validation (keep blank for leave one out)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
