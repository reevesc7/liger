import argparse
from pathlib import Path
from dataclasses import dataclass, fields
import json
import pandas as pd
from liger import smallville
from liger.surveying.openai import OpenAISurveyor
from liger.embedding.openai import OpenAIEmbedder
from liger.embedding.sentencetrans import STEmbedder
from liger import transforms as tfs


@dataclass(slots=True)
class Data:
    prompts: pd.Series | None = None
    responses: pd.DataFrame | None = None
    functionals: pd.DataFrame | None = None
    embeddings_ai: pd.DataFrame | None = None
    embeddings_st: pd.DataFrame | None = None


@dataclass(slots=True)
class DataCategories:
    prompts: bool = False
    responses: bool = False
    functionals: bool = False
    embeddings_ai: bool = False
    embeddings_st: bool = False


@dataclass(slots=True)
class FilePaths:
    log: Path
    prompts: Path
    responses: Path
    functionals: Path
    embeddings_ai: Path
    embeddings_st: Path
    full: Path


@dataclass(slots=True)
class PromptConfig:
    starts: dict[str, int]
    ends: dict[str, int]
    whitelist: set[str]
    blacklist: set[str]
    alter_replaced: str
    alter_replacement: str
    prepend: str


@dataclass(slots=True)
class SurveyConfig:
    model: str
    response_seed: str
    allowed_responses: set[str]
    temperature: float


@dataclass(slots=True)
class Config:
    include: DataCategories
    retrieve: DataCategories
    save: DataCategories
    paths: FilePaths
    prompts: PromptConfig
    survey: SurveyConfig
    ai_embedding_model: str
    st_embedding_model: str


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
    output = Path(cfg["names"]["dataset"])
    return Config(
        include=DataCategories(**cfg["include"]),
        retrieve=DataCategories(**cfg["try_retrieve"]),
        save=DataCategories(**cfg["save"]),
        paths=FilePaths(
            log=(output / cfg["names"]["log"]).with_suffix(".txt"),
            prompts=(output / cfg["names"]["prompts"]).with_suffix(".csv"),
            responses=(output / cfg["names"]["responses"]).with_suffix(".csv"),
            functionals=(output / cfg["names"]["functionals"]).with_suffix(".csv"),
            embeddings_ai=(output / cfg["names"]["embeddings_ai"]).with_suffix(".csv"),
            embeddings_st=(output / cfg["names"]["embeddings_st"]).with_suffix(".csv"),
            full=(output / cfg["names"]["full"]).with_suffix(".csv"),
        ),
        prompts=PromptConfig(**cfg["prompts"]),
        survey=SurveyConfig(**cfg["surveying"]),
        ai_embedding_model=cfg["ai_embedding_model"],
        st_embedding_model=cfg["st_embedding_model"],
    )


def get_prompts(cfg: Config, data: Data) -> pd.Series:
    if data.prompts is not None:
        return data.prompts
    if cfg.retrieve.prompts and cfg.paths.prompts.exists():
        prompts = pd.read_csv(cfg.paths.prompts).squeeze(axis=1)
        if not isinstance(prompts, pd.Series):
            raise TypeError(f"Prompts read from {prompts} do not form a pandas.Series")
    else:
        prompts = smallville.get_logged_prompts(
            cfg.paths.log,
            cfg.prompts.starts,
            cfg.prompts.ends,
            cfg.prompts.whitelist,
            cfg.prompts.blacklist,
        )
    prompt_modder = OpenAIEmbedder("")
    if cfg.prompts.alter_replaced != "":
        prompts = prompt_modder.alter_strings(
            prompts,
            cfg.prompts.alter_replaced,
            cfg.prompts.alter_replacement,
        )
    if cfg.prompts.prepend != "":
        prompts = prompt_modder.prepend_to_strings(prompts, cfg.prompts.prepend)
    prompts.name = "prompt"
    return prompts


# def get_responses(cfg: Config, data: Data) -> pd.DataFrame:
#     if data.responses is not None:
#         return data.responses
#     if cfg.retrieve.responses and cfg.paths.responses.exists():
#         return pd.read_csv(cfg.paths.responses)
#     return OpenAISurveyor(cfg.survey.model).log_probs_survey(
#         get_prompts(cfg, data),
#         cfg.survey.response_seed,
#         cfg.survey.allowed_responses,
#     )


def get_responses(cfg: Config, data: Data) -> pd.Series:
    # if data.responses is not None:
    #     return data.responses
    # if cfg.retrieve.responses and cfg.paths.responses.exists():
    #     return pd.read_csv(cfg.paths.responses)
    return OpenAISurveyor(cfg.survey.model).survey(
        get_prompts(cfg, data),
        1,
        temperature=0.0,
    )


def get_functionals(cfg: Config, data: Data) -> pd.DataFrame:
    if data.functionals is not None:
        return data.functionals
    if cfg.retrieve.functionals and cfg.paths.functionals.exists():
        return pd.read_csv(cfg.paths.functionals)
    logprobs = get_responses(cfg, data)
    return pd.concat([
        tfs.apply_logprobs_mode(logprobs, temperature=cfg.survey.temperature),
        tfs.apply_logprobs_mean(logprobs, temperature=cfg.survey.temperature),
        tfs.apply_logprobs_variance(logprobs, temperature=cfg.survey.temperature),
        tfs.apply_logprobs_std_dev(logprobs, temperature=cfg.survey.temperature),
    ], axis=1)


def get_embeddings_ai(cfg: Config, data: Data) -> pd.DataFrame:
    if data.embeddings_ai is not None:
        return data.embeddings_ai
    if cfg.retrieve.embeddings_ai and cfg.paths.embeddings_ai.exists():
        return pd.read_csv(cfg.paths.embeddings_ai)
    return OpenAIEmbedder(cfg.ai_embedding_model).embed(get_prompts(cfg, data))


def get_embeddings_st(cfg: Config, data: Data) -> pd.DataFrame:
    if data.embeddings_st is not None:
        return data.embeddings_st
    if cfg.retrieve.embeddings_st and cfg.paths.embeddings_st.exists():
        return pd.read_csv(cfg.paths.embeddings_st)
    return STEmbedder(cfg.st_embedding_model).embed(get_prompts(cfg, data))


def main():
    argparser = init_argparser()
    cfg_file = parse_args(argparser)
    cfg = read_config(cfg_file)
    data = Data()
    if cfg.include.prompts:
        data.prompts = get_prompts(cfg, data)
        if cfg.save.prompts:
            data.prompts.to_csv(cfg.paths.prompts, index=False)
    if cfg.include.responses:
        get_responses(cfg, data).to_csv(cfg.paths.responses, index=False)
        # data.responses = get_responses(cfg, data)
        # if cfg.save.responses:
        #     data.responses.to_csv(cfg.paths.responses, index=False)
    if cfg.include.functionals:
        data.functionals = get_functionals(cfg, data)
        if cfg.save.functionals:
            data.functionals.to_csv(cfg.paths.functionals, index=False)
    if cfg.include.embeddings_ai:
        data.embeddings_ai = get_embeddings_ai(cfg, data)
        if cfg.save.embeddings_ai:
            data.embeddings_ai.to_csv(cfg.paths.embeddings_ai, index=False)
    if cfg.include.embeddings_st:
        data.embeddings_st = get_embeddings_st(cfg, data)
        if cfg.save.embeddings_st:
            data.embeddings_st.to_csv(cfg.paths.embeddings_st, index=False)
    pd.concat([
        getattr(data, field.name)
        for field in fields(data)
    ], axis=1).to_csv(cfg.paths.full, index=False)


if __name__ == "__main__":
    main()

