# from sys import platform
from argparse import ArgumentParser, Namespace
from pipelines.tpot import TpotPipeline
# if platform == 'win32':
#     from os import environ
#     environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


PICKLE = "/Pickles/"

def main():
    args = get_args()

    data_name = TpotPipeline.get_data_name(args.data)
    id = TpotPipeline.get_id(args.ngens, args.popsize, args.tpotrs, args.regression, args.notrees, args.noxg)
    pickle_file = TpotPipeline.find_pickle(data_name, id)
    if pickle_file:
        pipeline = TpotPipeline.from_pickle(pickle_file)
        # pipeline.set_warm_start(True)
    else:
        pipeline = TpotPipeline(
            args.data,
            args.regression,
            args.ngens,
            args.popsize,
            args.tpotrs,
            [i+args.sevalrs for i in range(args.nevalrs)],
            args.notrees,
            args.noxg,
            args.cutoffmins,
            args.nsplits,
        )
    pipeline.run_1_gen()


def get_args() -> Namespace:
    parser = ArgumentParser(description="Script that runs tpot fitting and evaluation")
    parser.add_argument("--data", type=str, required=True, help="training data file path")      #"Data/human_size_rating_1_1.csv"
    parser.add_argument("--regression", action="store_true", dest="regression", help="run regression (not classification)")
    parser.add_argument("--ngens", type=int, required=False, default=5, help="number of tpot fitting generations")
    parser.add_argument("--popsize", type=int, required=False, default=50, help="tpot fitting population size")
    parser.add_argument("--tpotrs", type=int, required=False, default=0, help="tpot random state")
    parser.add_argument("--sevalrs", type=int, required=False, default=0, help="eval random state start")
    parser.add_argument("--nevalrs", type=int, required=False, default=1, help="number of eval random states to test")
    parser.add_argument("--notrees", action="store_true", dest="notrees", help="whether RandomForest and Tree models should be excluded")
    parser.add_argument("--noxg", action="store_true", dest="noxg", help="whether XGBoost models should be excluded")
    parser.add_argument("--cutoffmins", type=int, required=False, default=None, help="minutes at which tpot should terminate fitting early")
    parser.add_argument("--nsplits", type=int, required=False, default=None, help="number of splits for K-fold validation (keep blank for leave one out)")
    return parser.parse_args()


if __name__ == "__main__":
    main()

