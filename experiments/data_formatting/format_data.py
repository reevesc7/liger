import pandas as pd
import config as cfg
import get_prompts
import get_responses
import get_functionals
import get_embeddings_ai
import get_embeddings_st


def format_data() -> pd.DataFrame:
    data: list[pd.DataFrame | pd.Series] = []
    if cfg.PROMPTS_OP != cfg.Op.SKIP:
        data.append(get_prompts.prompts())
        if cfg.SAVE_INTERMEDIATES:
            data[-1].to_csv(cfg.PROMPTS_FILE, index=False)
    if cfg.RESPONSES_OP != cfg.Op.SKIP:
        data.append(get_responses.responses())
        if cfg.SAVE_INTERMEDIATES:
            data[-1].to_csv(cfg.RESPONSES_FILE, index=False)
    if cfg.FUNCTIONALS_OP != cfg.Op.SKIP:
        data.append(get_functionals.functionals())
        if cfg.SAVE_INTERMEDIATES:
            data[-1].to_csv(cfg.FUNCTIONALS_FILE, index=False)
    if cfg.EMBEDDINGS_AI_OP != cfg.Op.SKIP:
        data.append(get_embeddings_ai.embeddings_ai())
        if cfg.SAVE_INTERMEDIATES:
            data[-1].to_csv(cfg.EMBEDDINGS_AI_FILE, index=False)
    if cfg.EMBEDDINGS_ST_OP != cfg.Op.SKIP:
        data.append(get_embeddings_st.embeddings_st())
        if cfg.SAVE_INTERMEDIATES:
            data[-1].to_csv(cfg.EMBEDDINGS_ST_FILE, index=False)
    return pd.concat(data, axis=1)


def main():
    format_data().to_csv(cfg.DATASET_FILE, index=False)


if __name__ == "__main__":
    main()

