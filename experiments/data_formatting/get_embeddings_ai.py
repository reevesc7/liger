import pandas as pd
from liger.embedding.openai import OpenAIEmbedder
import constants as cnst
import get_prompts


def embeddings_ai() -> pd.DataFrame:
    prompts = get_prompts.prompts()
    if cnst.EMBEDDINGS_AI_OP == cnst.Op.MAKE:
        embedder = OpenAIEmbedder(cnst.AI_EMBED_MODEL)
        return embedder.embed(prompts)
    return pd.read_csv(cnst.EMBEDDINGS_AI_FILE)


def main():
    embeddings_ai().to_csv(cnst.EMBEDDINGS_AI_FILE, index=False)


if __name__ == "__main__":
    main()

