import ast
import numpy as np
import pandas as pd
from liger.dataset import Dataset
from liger import smallville
from liger.surveying.openai import OpenAISurveyor
from liger.embedding.openai import OpenAIEmbedder
from liger.embedding.sentencetrans import STEmbedder


LOG_FILE = "../../../generative_agents/response_logs/simulation_test_013_2024-10-24.txt"
RESPONSE_SEED = "{\"output\": \""
ALLOWED_TOKENS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
AI_SURVEY = "gpt-3.5-turbo"
AI_EMBED = "text-embedding-3-small"
ST_EMBED = "all-mpnet-base-v2"


def get_probs(prompts: pd.Series) -> pd.DataFrame:
    surveyor = OpenAISurveyor(AI_SURVEY)
    return surveyor.probs_survey(
        prompts,
        RESPONSE_SEED,
        ALLOWED_TOKENS,
        allow_dupes=True
    )


def get_ai_embeddings(prompts: pd.Series) -> pd.DataFrame:
    embedder = OpenAIEmbedder(AI_EMBED)
    return embedder.embed(prompts)


def get_st_embeddings(prompts: pd.Series) -> pd.DataFrame:
    embedder = STEmbedder(ST_EMBED)
    return embedder.embed(prompts)


def main():
    #prompts = smallville.get_logged_prompts(LOG_FILE, "rate the likely poignancy")
    #prompts.to_csv("dataset_test_prompts.csv", index=False)
    prompts = pd.read_csv("dataset_test_prompts.csv")

    #probs = get_probs(prompts)
    #probs.to_csv("dataset_test_probs.csv", index=False)
    probs = pd.read_csv("dataset_test_probs.csv")
    probs.columns = pd.Index(f"prob_{col}" for col in probs.columns)

    #functionals = OpenAISurveyor.functionals(probs)
    #functionals.to_csv("dataset_test_functionals.csv", index=False)
    functionals = pd.read_csv("dataset_test_functionals.csv")

    #ai_embeddings = get_ai_embeddings(prompts)
    #ai_embeddings.to_csv("dataset_test_embed_ai.csv", index=False)
    ai_embeddings = pd.read_csv("dataset_test_embed_ai.csv")

    #st_embeddings = get_st_embeddings(prompts)
    #st_embeddings.to_csv("dataset_test_embed_st.csv", index=False)
    st_embeddings = pd.read_csv("dataset_test_embed_st.csv")

    df = pd.concat((prompts, probs, functionals, ai_embeddings, st_embeddings), axis=1)
    #df = pd.DataFrame({
    #    "text": prompts,
    #    AI_EMBED: ai_embeddings,
    #    ST_EMBED: st_embeddings,
    #    "probs": probs,
    #    "mean": functionals["mean"],
    #    "mode": functionals["mode"],
    #    "std_dev": functionals["std_dev"],
    #})
    print(df)
    df.to_csv("dataset_test.csv", index=False)


if __name__ == "__main__":
    main()

