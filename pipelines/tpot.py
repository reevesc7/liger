from os import makedirs, remove
from os.path import isdir, isfile
from time import sleep
import numpy as np
import pandas as pd
from ..dataset import Dataset
from ..training_testing import kfold_predict
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import classifier_config_dict, regressor_config_dict
from sklearn.model_selection import KFold
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score
import pickle
import dill
from deap import base, creator, gp


OUTPUT= "Outputs/tpot/"
PICKLE = "Pickles/"
PERFORMANCE = "performance.csv"
RATINGS = "ratings.csv"


class TpotPipeline:

    def __init__(
        self,
        data_file: str,
        regression: bool,
        target_gens: int,
        pop_size: int,
        tpot_random_state: int,
        eval_random_states: list[int],
        no_trees: bool = False,
        no_xg: bool = False,
        cutoff_mins: int | None = None,
        n_splits: int | None = None,
    ):
        self.dataset = Dataset.from_csv(data_file)
        self.data_name = TpotPipeline.get_data_name(data_file)
        self.target_gens = target_gens
        self.complete_gens = 0
        self.pop_size = pop_size
        self.tpot_random_state = tpot_random_state
        self.eval_random_states = eval_random_states
        self.no_trees = no_trees
        self.no_xg = no_xg
        self.cutoff_mins = cutoff_mins

        if not regression and self.dataset.y.dtype == np.float64:
            print("WARNING: Float data detected, using regression instead of classification.", flush=True)
            regression = True
        self.regression = regression
        if n_splits:
            self.n_splits = n_splits
        else:
            self.n_splits = self.dataset.X.shape[0]

        # NOTE: Whether or not regression was set to true by detection of floats,
        #       regression in the ID will be what the arguments specified (reg/clas).
        self.id = TpotPipeline.get_id(target_gens, pop_size, tpot_random_state, regression, no_trees, no_xg)
        self.tpot = self.tpot_init()

        self.output_dir = OUTPUT + self.data_name + "/"
        self.pickle_dir = PICKLE + self.data_name + "/"


    @classmethod
    def get_data_name(cls, data_file: str) -> str:
        return data_file.rsplit("/")[-1].split(".")[0]


    @classmethod
    def get_id(cls, target_gens: int, pop_size: int, tpot_random_state: int, regression: bool, no_trees: bool, no_xg: bool) -> str:
        return "gens_" + str(target_gens) + "_popsize_" + \
            str(pop_size) + "_tpotrs_" + str(tpot_random_state) + \
                "_reg" if regression else "_clas" + \
                    "_notrees" if no_trees else "" + "_noxg" if no_xg else ""


    @classmethod
    def find_pickle(cls, data_name: str, id: str) -> str | None:
        pickle_name = PICKLE + data_name + "/" + id + ".pkl"
        if not isdir(PICKLE):
            return None
        if isfile(pickle_name):
            return pickle_name
        return None


    @classmethod
    def from_pickle(cls, pickle_file: str) -> 'TpotPipeline':
        with open(pickle_file, 'rb') as f:
            return TpotPipeline.from_bytes(f.read())


    @classmethod
    def from_bytes(cls, serialization: bytes) -> 'TpotPipeline':
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMulti,
            statistics=dict,
        )
        return dill.loads(serialization)
        # return pickle.loads(serialization)


    def to_pickle(self) -> None:
        with open(self.pickle_dir + self.id + ".pkl", "wb") as f:
            f.write(self.to_bytes())


    def to_bytes(self) -> bytes:
        self.tpot._pbar = None
        return dill.dumps(self)
        # return pickle.dumps(self)


    def set_warm_start(self, warm_start: bool) -> None:
        self.tpot.warm_start = warm_start


    def run_1_gen(self) -> None:

        # Save prep
        self.save_prep()
        print("\nRUNNING PIPELINE:", self.id, flush=True)
        print("GENERATION:", self.complete_gens + 1, flush=True)
        # print(self.dataset.X)
        # print(self.dataset.y)
        # print("X array analysis")
        # print(type(self.dataset.X))
        # print(self.dataset.X.dtype)
        # print("subarray analysis")
        # print(type(self.dataset.X[0]))
        # print(self.dataset.X[0].dtype)
        # print("subsubarray analysis")
        # print(type(self.dataset.X[0][0]))
        # print(self.dataset.X[0][0].dtype)
        # for prompt in self.dataset.X:
        #     try:
        #         print(np.any(np.isnan(prompt)))
        #     except:
        #         print(prompt)
        # print(np.any(np.isnan(self.dataset.X)))
        if self.complete_gens < self.target_gens:
            self.tpot.fit(self.dataset.X, self.dataset.y)
            self.complete_gens += 1
        if self.complete_gens >= self.target_gens:
            self.tpot.export(self.output_dir + self.id + ".py")
            self.evaluate()
            self.to_pickle()
            print("\nRUN COMPLETE")
            return
        self.to_pickle()
        print("\nRUN INCOMPLETE")
        return


    def save_prep(self) -> None:
        if not isdir(self.pickle_dir):
            makedirs(self.pickle_dir)
        if not isdir(self.output_dir):
            makedirs(self.output_dir)
        if not isfile(self.output_dir + PERFORMANCE):
            pd.DataFrame(columns=np.array([
                "regression",
                "n_gens",
                "pop_size",
                "trees_allowed",
                "tpot_random_state",
                "eval_random_state",
                "training_score",
                "n_splits",
                "KFold_R2"
            ])).to_csv(self.output_dir + PERFORMANCE, index=False)
        if not isfile(self.output_dir + RATINGS):
            pd.DataFrame(index=self.dataset.y).to_csv(self.output_dir + RATINGS)


    def tpot_init(self):
        if self.regression:
            custom_config = regressor_config_dict
        else:
            custom_config = classifier_config_dict
        if self.no_trees:
            custom_config = {key: value for key, value in custom_config.items() if 'RandomForest' not in key and 'Tree' not in key}
        if self.no_xg:
            custom_config = {key: value for key, value in custom_config.items() if 'XG' not in key}
        if self.regression:
            tpot = TPOTRegressor(
                config_dict=custom_config,
                generations=1,
                population_size=self.pop_size,
                verbosity=2,
                random_state=self.tpot_random_state,
                max_time_mins=self.cutoff_mins,
                warm_start=True,
            )
        else:
            tpot = TPOTClassifier(
                config_dict=custom_config,
                generations=1,
                population_size=self.pop_size,
                verbosity=2,
                random_state=self.tpot_random_state,
                max_time_mins=self.cutoff_mins,
                warm_start=True,
            )
        return tpot


    def evaluate(self) -> None:
        training_score = self.tpot.score(self.dataset.X, self.dataset.y)
        for eval_random_state in self.eval_random_states:
            kfold_predictions, kfold_r2 = self.tpot_test(eval_random_state)
            self.wait_for_free_csv()
            with open(self.output_dir + "unsafe.txt", 'w') as _:
                pass
            ratings = pd.read_csv(self.output_dir + RATINGS)
            ratings.insert(ratings.shape[1], self.id, kfold_predictions, True)
            ratings.to_csv(self.output_dir + RATINGS, index=False)
            performances = pd.read_csv(self.output_dir + PERFORMANCE)
            performances.loc[performances.shape[0]] = pd.Series(
                {
                    'regression': self.regression,
                    'n_gens': self.target_gens,
                    'pop_size': self.pop_size,
                    'trees_allowed': not self.no_trees,
                    'tpot_random_state': self.tpot_random_state,
                    'eval_random_state': eval_random_state,
                    'training_score': training_score,
                    'n_splits': self.n_splits,
                    'KFold_R2': kfold_r2
                }
            )
            performances.to_csv(self.output_dir + PERFORMANCE, index=False)
            remove(self.output_dir + "unsafe.txt")


    def tpot_test(self, eval_random_state: int) -> tuple[np.ndarray, float]:
        set_param_recursive(self.tpot.fitted_pipeline_.steps, 'random_state', eval_random_state)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=eval_random_state)
        kfold_predictions = kfold_predict(self.tpot.fitted_pipeline_, kfold, self.dataset)
        kfold_r2 = float(r2_score(kfold_predictions, self.dataset.y))
        return kfold_predictions, kfold_r2


    def wait_for_free_csv(self) -> None:
        loop_count = 0
        while isfile(self.output_dir + "unsafe.txt"):
            sleep(0.1)
            loop_count += 1
            if loop_count > 100:
                print("WARNING: Output csv possibly stuck marked as not safe for writing.")
                return

