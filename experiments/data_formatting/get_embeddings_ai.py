import pandas as pd
from liger.embedding.openai import OpenAIEmbedder
import config as cfg
import get_prompts


def embeddings_ai() -> pd.DataFrame:
    if cfg.EMBEDDINGS_AI_OP != cfg.Op.MAKE:
        return pd.read_csv(cfg.EMBEDDINGS_AI_FILE)
    prompts = get_prompts.prompts()
    embedder = OpenAIEmbedder(cfg.AI_EMBED_MODEL)
    return embedder.embed(prompts)


def main():
    embeddings_ai().to_csv(cfg.EMBEDDINGS_AI_FILE, index=False)


if __name__ == "__main__":
    main()

