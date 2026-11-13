import hashlib
from pathlib import Path
from ConfigSpace import ConfigurationSpace as CfgSpace, Categorical, Float, Integer
from sklearn import decomposition, ensemble, kernel_approximation, metrics
from tpot.search_spaces import SearchSpace
from tpot.config import get_search_space
from tpot.search_spaces import nodes as nd, pipelines as pl
from liger import loaders, pathers, sklearn
from liger.context import Context
from liger import tpot as lgtpot


OUTPUT_DIR = "example_outputs"
DATA_PATH = "../example_data.csv"
X_COLS = "^all-mpnet-base-v2"
Y_COLS = "^mean$"

POP_SIZE = 10
RANDOM_STATE = 1

CV = 4
EVAL_TIME = 1.0

SEGMENT_GENS = 3
TOTAL_GENS = 10
EARLY_STOP = 3

SIMPLE_ROOTS = (
    "Lasso",
    "LassoLars",
)
SIMPLE_INNERS = (
    "FeatureAgglomeration",
    "FastICA",
    "PCA",
    "RBFSampler",
    "Binarizer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "StandardScaler",
)

BAD_NODE = ValueError("Received invalid node name")


def _get_non_null_search_space(
    name: str,
    params: lgtpot.SearchSpaceParams,
) -> SearchSpace:
    space = get_search_space(name, **params)
    if space is None:
        raise ValueError("Received invalid node name")
    return space


def _root_space(params: lgtpot.SearchSpaceParams) -> pl.ChoicePipeline:
    spaces = [_get_non_null_search_space(name, params) for name in SIMPLE_ROOTS]
    knn_space = _get_non_null_search_space("KNeighborsRegressor", params)
    bag_knn_space = pl.WrapperPipeline(
        ensemble.BaggingRegressor,
        CfgSpace(space=[
            Integer("n_estimators", (3, 100)),
            Float("max_samples", (0.1, 1.0)),
            Float("max_features", (0.1, 1.0)),
            Categorical("bootstrap", [True, False]),
            Categorical("bootstrap_features", [True, False]),
            Categorical("random_state", [params["random_state"]]),
        ]),
        estimator_search_space=knn_space,
    )
    spaces.append(bag_knn_space)
    spaces.append(knn_space)
    return pl.ChoicePipeline(spaces)


def _inner_space(params: lgtpot.SearchSpaceParams) -> pl.ChoicePipeline:
    spaces = [_get_non_null_search_space(name, params) for name in SIMPLE_INNERS]
    spaces.append(nd.EstimatorNode(
        decomposition.TruncatedSVD,
        CfgSpace(space=[
            Integer("n_components", (1, 200)),
            Categorical("algorithm", ("arpack", "randomized")),
            Integer("n_iter", (2, 8)),
            Integer("n_oversamples", (1, 30)),
            Categorical("random_state", [params["random_state"]]),
        ]),
    ))
    spaces.append(nd.EstimatorNode(
        kernel_approximation.Nystroem,
        CfgSpace(space=[
            Categorical("kernel", [
                "rbf",
                "cosine",
                "laplacian",
                "polynomial",
                "poly",
                "linear",
                "sigmoid",
            ]),
            Float("gamma", (0.0, 1.0)),
            Integer("n_components", (1, params["n_features"])),
            Categorical("random_state", [params["random_state"]]),
        ]),
    ))
    spaces.append(nd.EstimatorNode(sklearn.LgPassthrough, CfgSpace()))
    return pl.ChoicePipeline(spaces)


def _search_space(params: lgtpot.SearchSpaceParams) -> pl.GraphSearchPipeline:
    return pl.GraphSearchPipeline(
        _root_space(params),
        inner_search_space=_inner_space(params),
    )


def _init_objectives() -> list[lgtpot.Objective]:
    obj: list[lgtpot.Objective] = []
    obj.append(lgtpot.Objective(
        metrics.get_scorer("neg_mean_squared_error"),
        weight=1.0,
        early_stop_tol=0.001,
    ))
    obj.append(lgtpot.Objective(
        lgtpot.objectives.score_complexity,
        weight=-1.0,
        early_stop_tol=0.001,
    ))
    return obj


def build(context: Context) -> lgtpot.TPOTManager:
    output_dir = context.once("output_dir", lambda: str(pathers.datetime_subpath(
        OUTPUT_DIR,
        "liger_tpot_{}",
        absolute=True,
    )))
    data_path = context.once("data_path", lambda: str(Path(DATA_PATH).resolve()))
    context.pin("data_hash", str(hashlib.md5(Path(data_path).read_bytes()).hexdigest()))
    return lgtpot.TPOTManager(
        output_dir,
        x=lambda: loaders.data_from_csv(data_path, columns=X_COLS),
        y=lambda: loaders.data_from_csv(data_path, columns=X_COLS),
        search_space=_search_space,
        classification=False,
        objectives=_init_objectives(),
        evolution_params=lgtpot.EvolutionParams(
            population_size=POP_SIZE,
            random_state=RANDOM_STATE,
        ),
        eval_params=lgtpot.EvalParams(
            cv=CV,
            eval_time=EVAL_TIME,
        ),
        end_params=lgtpot.EndParams(
            segment_generations=SEGMENT_GENS,
            total_generations=TOTAL_GENS,
            early_stop=EARLY_STOP,
        ),
        verbose=4,
    )
