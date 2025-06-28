import pandas as pd
import constants as cnst
import get_prompts
import get_responses
import get_functionals
import get_embeddings_ai
import get_embeddings_st


def format_data() -> pd.DataFrame:
    data: list[pd.DataFrame | pd.Series] = []
    if cnst.PROMPTS_OP != cnst.Op.SKIP:
        data.append(get_prompts.prompts())
        if cnst.SAVE_INTERMEDIATES:
            data[-1].to_csv(cnst.PROMPTS_FILE, index=False)
    if cnst.RESPONSES_OP != cnst.Op.SKIP:
        data.append(get_responses.responses())
        if cnst.SAVE_INTERMEDIATES:
            data[-1].to_csv(cnst.RESPONSES_FILE, index=False)
    if cnst.FUNCTIONALS_OP != cnst.Op.SKIP:
        data.append(get_functionals.functionals())
        if cnst.SAVE_INTERMEDIATES:
            data[-1].to_csv(cnst.FUNCTIONALS_FILE, index=False)
    if cnst.EMBEDDINGS_AI_OP != cnst.Op.SKIP:
        data.append(get_embeddings_ai.embeddings_ai())
        if cnst.SAVE_INTERMEDIATES:
            data[-1].to_csv(cnst.EMBEDDINGS_AI_FILE, index=False)
    if cnst.EMBEDDINGS_ST_OP != cnst.Op.SKIP:
        data.append(get_embeddings_st.embeddings_st())
        if cnst.SAVE_INTERMEDIATES:
            data[-1].to_csv(cnst.EMBEDDINGS_ST_FILE, index=False)
    return pd.concat(data, axis=1)


def main():
    format_data().to_csv(cnst.DATASET_FILE, index=False)


if __name__ == "__main__":
    main()

