import pandas as pd
from liger.embedding.sentencetrans import STEmbedder
import config as cfg
import get_prompts


def embeddings_st() -> pd.DataFrame:
    if cfg.EMBEDDINGS_ST_OP != cfg.Op.MAKE:
        return pd.read_csv(cfg.EMBEDDINGS_ST_FILE)
    prompts = get_prompts.prompts()
    embedder = STEmbedder(cfg.ST_EMBED_MODEL)
    return embedder.embed(prompts)


def main():
    embeddings_st().to_csv(cfg.EMBEDDINGS_ST_FILE, index=False)


if __name__ == "__main__":
    main()

