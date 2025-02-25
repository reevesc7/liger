from typing import Any
import sys
from os import makedirs, remove
from os.path import isdir, isfile
import json
from random import randint
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from ..dataset import Dataset
from ..training_testing import kfold_predict
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import classifier_config_dict, regressor_config_dict
from sklearn.model_selection import KFold
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score
import dill
from deap import base, creator, gp


OUTPUT= "Outputs/TPOT/"
PICKLE = "Pickles/"
IN_PROGRESS = "InProgress/"
PERFORMANCE = "performance.csv"
RATINGS = "ratings.csv"
TPOT_ATTRS = [
    "fitted_pipeline_",
    "pareto_front_fitted_pipelines_",
    "evaluated_individuals_",
]
REG_CLASS_OVERLAP = {
    "sklearn.preprocessing.Binarizer",
    "sklearn.decomposition.FastICA",
    "sklearn.cluster.FeatureAgglomeration",
    "sklearn.preprocessing.MaxAbsScaler",
    "sklearn.preprocessing.MinMaxScaler",
    "sklearn.preprocessing.Normalizer",
    "sklearn.kernel_approximation.Nystroem",
    "sklearn.decomposition.PCA",
    "sklearn.preprocessing.PolynomialFeatures",
    "sklearn.kernel_approximation.RBFSampler",
    "sklearn.preprocessing.RobustScaler",
    "sklearn.preprocessing.StandardScaler",
    "tpot.builtins.ZeroCount",
    "tpot.builtins.OneHotEncoder",
    "sklearn.feature_selection.SelectFwe",
    "sklearn.feature_selection.SelectPercentile",
    "sklearn.feature_selection.VarianceThreshold",
    "sklearn.feature_selection.SelectFromModel",
}
PIPELINE_PARAM_KEYS = {
    "config_name",
    "data_name",
    "regression",
    "target_gens",
    "eval_random_states",
    "n_splits",
    "slurm_id",
    "id",
}
TPOT_PARAM_KEYS = {
    "population_size",
    "offspring_size",
    "generations",
    "mutation_rate",
    "crossover_rate",
    "scoring",
    "cv",
    "subsample",
    "n_jobs",
    "max_time_mins",
    "max_eval_time_mins",
    "periodic_checkpoint_folder",
    "early_stop",
    "config_dict",
    "template",
    "warm_start",
    "memory",
    "use_dask",
    "verbosity",
    "disable_update_check",
    "random_state",
    "log_file",
}
TPOT_ATTR_KEYS = {
    "tree_structure",
    "evaluated_individuals_",
    "fitted_pipeline_"
    "log_file_",
    "pareto_front_fitted_pipelines_",
}


class TPOTPipeline:

    def __init__(
        self,
        config_file: str,
        data_file: str,
        target_gens: int | None = None,
        population_size: int | None = None,
        tpot_random_state: int | None = None,
        eval_random_states: list[int] | None = None,
        max_time_mins: int | None = None,
        n_splits: int | None = None,
        slurm_id: int | None = None,
        id: str | None = None,
    ) -> None:
        self.config_name = TPOTPipeline.get_filename(config_file)
        self.dataset = Dataset.from_csv(data_file)
        self.data_name = TPOTPipeline.get_filename(data_file)
        self.complete_gens = 0
        self.slurm_id = slurm_id

        # Load config file
        with open(config_file) as f:
            config = dict(json.load(f))
        pipeline_parameters = config.get("pipeline_parameters")
        if pipeline_parameters is None:
            pipeline_parameters = {}
        tpot_parameters = config.get("tpot_parameters")
        if tpot_parameters is None:
            tpot_parameters = {}

        # Record start time
        dt = datetime.now(timezone.utc)
        start_time = dt.strftime("%Y-%m-%d_%H-%M-%S.%f")

        # Require regression if there's float data
        if self.dataset.y.dtype == np.float64:
            require_regression = True
        else:
            require_regression = False

        # Set and vet the TPOT config dictionary; determine if regression will be used
        config_dict = tpot_parameters.get("config_dict")
        if isinstance(config_dict, dict):
            self.regression = self.check_config_dict(config_dict, require_regression)
        elif config_dict is None:
            config_dict = regressor_config_dict
            self.regression = True
        else:
            raise TypeError(f"\"config_dict\" field of {self.config_name} is not None or of type dict")

        # Set parameters
        self.target_gens = TPOTPipeline.use_first(
            target_gens,
            pipeline_parameters.get("target_gens"),
            10,
        )
        population_size = int(TPOTPipeline.use_first(
            population_size,
            tpot_parameters.get("population_size"),
            100,
        ))
        tpot_random_state = TPOTPipeline.use_first(
            tpot_random_state,
            tpot_parameters.get("random_state"),
            randint(0, 9999999999),
        )
        self.eval_random_states = TPOTPipeline.use_first(
            eval_random_states,
            pipeline_parameters.get("eval_random_states"),
            [0],
        )
        self.id = TPOTPipeline.use_first(
            id,
            pipeline_parameters.get("id"),
            start_time,
        )
        max_time_mins = TPOTPipeline.use_first(
            max_time_mins,
            tpot_parameters.get("max_time_mins"),
        )
        self.n_splits = TPOTPipeline.use_first(
            n_splits,
            pipeline_parameters.get("n_splits"),
            self.dataset.X.shape[0],
        )

        if self.regression:
            self.tpot = TPOTRegressor(
                config_dict=config_dict,
                generations=1,
                population_size=population_size,
                verbosity=2,
                random_state=tpot_random_state,
                max_time_mins=max_time_mins,
                warm_start=True,
            )
        else:
            self.tpot = TPOTClassifier(
                config_dict=config_dict,
                generations=1,
                population_size=population_size,
                verbosity=2,
                random_state=tpot_random_state,
                max_time_mins=max_time_mins,
                warm_start=True,
            )


        self.output_dir = OUTPUT + self.data_name + "/"
        self.pickle_dir = PICKLE + self.data_name + "/"
        self.inprogress_dir = IN_PROGRESS + self.data_name + "/"

        print("ID =", self.id, flush=True)
        print("TPOT random state =", tpot_random_state, flush=True)


    @classmethod
    def from_pickle(cls, pickle_file: str) -> 'TPOTPipeline':
        with open(pickle_file, "rb") as f:
            return TPOTPipeline.from_bytes(f.read())


    @classmethod
    def from_bytes(cls, serialization: bytes) -> 'TPOTPipeline':
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMulti,
            statistics=dict,
        )
        return dill.loads(serialization)


    @staticmethod
    def use_first(*args) -> Any:
        """Return the first arg which is not None."""
        for arg in args:
            if arg is not None:
                return arg
        return None


    @staticmethod
    def get_filename(data_file: str) -> str:
        return data_file.rsplit("/")[-1].split(".")[0]


    @staticmethod
    def get_id(target_gens: int, pop_size: int, tpot_random_state: int, regression: bool, no_trees: bool, no_xg: bool) -> str:
        return "gens_" + str(target_gens) + "_popsize_" + \
            str(pop_size) + "_tpotrs_" + str(tpot_random_state) + \
                "_reg" if regression else "_clas" + \
                    "_notrees" if no_trees else "" + "_noxg" if no_xg else ""


    @staticmethod
    def find_pickle(data_name: str, id: str) -> str | None:
        pickle_name = PICKLE + data_name + "/" + id + ".pkl"
        if not isdir(PICKLE):
            return None
        if isfile(pickle_name):
            return pickle_name
        return None


    @staticmethod
    def dict_everything(objec) -> str | None:
        if hasattr(objec, "__dict__"):
            return json.dumps(objec.__dict__, indent=4, default=TPOTPipeline.dict_everything)
        else:
            return None


    def check_config_dict(self, config_dict: dict, require_regression: bool) -> bool:
        reg_key = False
        clas_key = False
        for key in config_dict.keys():
            if key in REG_CLASS_OVERLAP:
                continue
            if key in regressor_config_dict:
                reg_key = True
            elif key in classifier_config_dict:
                clas_key = True
            else:
                raise KeyError(f"{key} is not a valid sklearn model")
        if clas_key and require_regression:
            raise TypeError(f"float values found in {self.data_name}, but {self.config_name} contains classifiers")
        if reg_key and clas_key:
            raise ValueError(f"{self.config_name} contains regressors and classifiers")
        return not clas_key


    def to_pickle(self) -> None:
        bytestring = self.to_bytes()
        with open(self.pickle_dir + self.id + ".pkl", "wb") as f:
            f.write(bytestring)


    def to_bytes(self) -> bytes:
        self.tpot._pbar = None
        try:
            return dill.dumps(self)
        except Exception as e:
            print("\n", e)
            sys.exit(1)


    def run_1_gen(self) -> None:

        # Save prep
        self.save_prep()
        self.in_progress()
        print("\nRUNNING PIPELINE:", self.id, flush=True)
        print("GENERATION:", self.complete_gens + 1, flush=True)
        if self.complete_gens < self.target_gens:
            self.tpot.fit(self.dataset.X, self.dataset.y)
            self.complete_gens += 1
        if self.complete_gens >= self.target_gens:
            self.tpot.export(self.output_dir + self.id + ".py")
            self.evaluate()
            self.to_pickle()
            self.not_in_progress()
            print("\nRUN COMPLETE")
            return
        self.to_pickle()
        self.not_in_progress()
        print("\nRUN INCOMPLETE")
        return


    def save_prep(self) -> None:
        if not isdir(self.pickle_dir):
            makedirs(self.pickle_dir)
        if not isdir(self.inprogress_dir):
            makedirs(self.inprogress_dir)
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


    def in_progress(self):
        with open(self.inprogress_dir + self.id + ".txt", "w") as f:
            f.writelines([
                "Start: UTC " + str(datetime.now(timezone.utc)),
                "\nGeneration: " + str(self.complete_gens + 1),
                "\nSLURM JOB ID: " + str(self.slurm_id),
            ])


    def not_in_progress(self) -> None:
        remove(self.inprogress_dir + self.id + ".txt")


    # def evaluate(self) -> None:
    #     training_score = self.tpot.score(self.dataset.X, self.dataset.y)
    #     for eval_random_state in self.eval_random_states:
    #         kfold_predictions, kfold_r2 = self.tpot_test(eval_random_state)
    #         self.wait_for_free_csv()
    #         with open(self.output_dir + "unsafe.txt", 'w') as _:
    #             pass
    #         ratings = pd.read_csv(self.output_dir + RATINGS)
    #         ratings.insert(ratings.shape[1], self.id, kfold_predictions, True)
    #         ratings.to_csv(self.output_dir + RATINGS, index=False)
    #         performances = pd.read_csv(self.output_dir + PERFORMANCE)
    #         performances.loc[performances.shape[0]] = pd.Series(
    #             {
    #                 'regression': self.regression,
    #                 'n_gens': self.target_gens,
    #                 'pop_size': self.pop_size,
    #                 'trees_allowed': not self.no_trees,
    #                 'tpot_random_state': self.tpot_random_state,
    #                 'eval_random_state': eval_random_state,
    #                 'training_score': training_score,
    #                 'n_splits': self.n_splits,
    #                 'KFold_R2': kfold_r2
    #             }
    #         )
    #         performances.to_csv(self.output_dir + PERFORMANCE, index=False)
    #         remove(self.output_dir + "unsafe.txt")


    def evaluate(self) -> None:
        training_score = self.tpot.score(self.dataset.X, self.dataset.y)
        ratings = {}
        scores = {}
        for eval_random_state in self.eval_random_states:
            kfold_predictions, kfold_r2 = self.tpot_test(eval_random_state)
            ratings[eval_random_state] = kfold_predictions
            scores[eval_random_state] = kfold_r2
        pipeline_parameters = {key: value for key, value in self.__dict__.items() if key in PIPELINE_PARAM_KEYS}
        tpot_parameters = {key: value for key, value in self.tpot.__dict__.items() if key in TPOT_PARAM_KEYS}
        tpot_attributes = {key: value for key, value in self.tpot.__dict__.items() if key in TPOT_ATTR_KEYS}
        pipeline_dict = {
            "pipeline_parameters": pipeline_parameters,
            "tpot_parameters": tpot_parameters,
            "tpot_attributes": tpot_attributes,
        }
        with open(self.output_dir + str(self.id) + "output.json", "w") as f:
            json.dump(pipeline_dict, f, default=TPOTPipeline.dict_everything)


    def tpot_test(self, eval_random_state: int) -> tuple[np.ndarray, float]:
        set_param_recursive(self.tpot.fitted_pipeline_.steps, 'random_state', eval_random_state)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=eval_random_state)
        kfold_predictions = kfold_predict(self.tpot.fitted_pipeline_, kfold, self.dataset)
        kfold_r2 = float(r2_score(kfold_predictions, self.dataset.y))
        return kfold_predictions, kfold_r2

