import pandas as pd
from liger.surveying.openai import OpenAISurveyor
import constants as cnst
import get_prompts


def responses() -> pd.DataFrame:
    if cnst.RESPONSES_OP != cnst.Op.MAKE:
        return pd.read_csv(cnst.RESPONSES_FILE)
    prompts = get_prompts.prompts()
    surveyor = OpenAISurveyor(cnst.AI_SURVEY_MODEL)
    return surveyor.probs_survey(prompts, cnst.RESPONSE_SEED, cnst.ALLOWED_TOKENS)


def main():
    responses().to_csv(cnst.RESPONSES_FILE, index=False)


if __name__ == "__main__":
    main()

