from os import makedirs, remove
from os.path import exists
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


OUTPUT= "/Outputs/tpot/"
PICKLE = "/Pickles/"
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
        self.data_name = data_file.rsplit('/')[-1].split('.')[0]
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
        self.id = "gens_" + str(target_gens) + "_popsize_" + \
            str(pop_size) + "_tpotrs_" + str(tpot_random_state) + \
                "_reg" if self.regression else "_clas" + \
                    "_notrees" if no_trees else "" + "_noxg" if no_xg else ""
        self.tpot = self.tpot_init()

        self.output_dir = OUTPUT + self.data_name + "/"
        self.pickle_dir = PICKLE + self.data_name + "/"


    @classmethod
    def from_bytes(cls, serialization: bytes) -> 'TpotPipeline':
        return pickle.loads(serialization)


    def to_bytes(self) -> bytes:
        return pickle.dumps(self)


    def run_1_gen(self) -> None:

        # Save prep
        self.save_prep()
        print("\nRunning pipeline:", self.id, flush=True)
        if self.complete_gens < self.target_gens:
            self.tpot.fit(self.dataset.X, self.dataset.y)
            self.complete_gens += 1
        if self.complete_gens >= self.target_gens:
            self.tpot.export(self.output_dir + self.id + ".py")
            self.evaluate()
            return
        self.freeze()
        return


    def save_prep(self) -> None:
        if not exists(self.pickle_dir):
            makedirs(self.pickle_dir)
        if not exists(self.output_dir):
            makedirs(self.output_dir)
        if not exists(self.output_dir + PERFORMANCE):
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
        if not exists(self.output_dir + RATINGS):
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
            )
        else:
            tpot = TPOTClassifier(
                config_dict=custom_config,
                generations=1,
                population_size=self.pop_size,
                verbosity=2,
                random_state=self.tpot_random_state,
                max_time_mins=self.cutoff_mins,
            )
        return tpot


    def evaluate(self) -> None:
        training_score = self.tpot.score(self.dataset.X, self.dataset.y)
        for eval_random_state in self.eval_random_states:
            kfold_predictions, kfold_r2 = self.tpot_test(eval_random_state)
            self.wait_for_free_csv(self.output_dir)
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


    def freeze(self) -> None:
        with open(self.pickle_dir + self.id + ".pkl", "wb") as f:
            f.write(self.to_bytes())


    def tpot_test(self, eval_random_state: int) -> tuple[np.ndarray, float]:
        set_param_recursive(self.tpot.fitted_pipeline_.steps, 'random_state', eval_random_state)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=eval_random_state)
        kfold_predictions = kfold_predict(self.tpot.fitted_pipeline_, kfold, self.dataset)
        kfold_r2 = float(r2_score(kfold_predictions, self.dataset.y))
        return kfold_predictions, kfold_r2


    def wait_for_free_csv(self, directory: str) -> None:
        loop_count = 0
        while exists(directory + "unsafe.txt"):
            sleep(0.1)
            loop_count += 1
            if loop_count > 100:
                print("WARNING: Output csv possibly stuck marked as not safe for writing.")
                return

