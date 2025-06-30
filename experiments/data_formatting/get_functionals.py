import pandas as pd
from liger.surveying.openai import OpenAISurveyor
import constants as cnst
import get_responses


def functionals() -> pd.DataFrame:
    if cnst.FUNCTIONALS_OP != cnst.Op.MAKE:
        return pd.read_csv(cnst.FUNCTIONALS_FILE)
    responses = get_responses.responses()
    return OpenAISurveyor.functionals(responses)


def main():
    functionals().to_csv(cnst.FUNCTIONALS_FILE, index=False)


if __name__ == "__main__":
    main()

