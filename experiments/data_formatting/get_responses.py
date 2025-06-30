import pandas as pd
from liger.surveying.openai import OpenAISurveyor
import config as cfg
import get_prompts


def responses() -> pd.DataFrame:
    if cfg.RESPONSES_OP != cfg.Op.MAKE:
        return pd.read_csv(cfg.RESPONSES_FILE)
    prompts = get_prompts.prompts()
    surveyor = OpenAISurveyor(cfg.AI_SURVEY_MODEL)
    return surveyor.probs_survey(prompts, cfg.RESPONSE_SEED, cfg.ALLOWED_TOKENS)


def main():
    responses().to_csv(cfg.RESPONSES_FILE, index=False)


if __name__ == "__main__":
    main()

