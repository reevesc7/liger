from typing import Self, TypeVar
import re
from dataclasses import dataclass
import pandas as pd


PROMPT_FILE = "smallville_417_maria/prompts.csv"
NO_AGENT_POIGNANCY = True


# """
# Here is a brief description of Maria Lopez. 
# Name: Maria Lopez
# Age: 21
# Innate traits: energetic, enthusiastic, inquisitive
# Learned traits: Maria Lopez is a student at Oak Hill College studying physics and a part time Twitch game streamer who loves to connect with people and explore new ideas.
# Currently: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
# Lifestyle: Maria Lopez goes to bed around 2am, awakes up around 9am, eats dinner around 6pm. She likes to hang out at Hobbs Cafe if it's before 6pm.
# Daily plan requirement: Maria Lopez spends at least 3 hours a day Twitch streaming or gaming.
# Current Date: Monday February 13
#
#
# On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for Maria Lopez.
#
# Event: Klaus Mueller values social interactions and is making plans to catch up with Isabella Rodriguez and attend a Valentine's Day party at the cafe, indicating a balance between academic and social life
# Rate (return a number between 1 to 10):
# """
# Output the response to the prompt above in json. The output should ONLY contain ONE integer value on the scale of 1 to 10.
# Example output json:
# {"output": "5"}


T = TypeVar("T")


@dataclass(slots=True)
class Prompt:
    agent: str
    poignancy: str
    event: str
    output_fmt: str

    def __new__(cls, prompt: str | Self) -> Self:
        if type(prompt) is cls:
            return prompt
        return object.__new__(cls)

    def __init__(self, prompt: str | Self) -> None:
        if isinstance(prompt, Prompt):
            return
        self.agent = self._some(re.search(r"(?s)^.*?Current Date.*?\n\n\n", prompt)).group(0)
        self.poignancy = self._some(re.search(r"(?s)On the scale.*?Event: ", prompt)).group(0)
        self.event = self._some(re.search(r"(?s)Event: (.*?\n)", prompt)).group(1)
        self.output_fmt = self._some(re.search(r"(?s)Rate \(.*$", prompt)).group(0)

    @staticmethod
    def _some(maybe: T | None) -> T:
        if maybe is None:
            raise TypeError(f"Received NoneType instead of type {T}")
        return maybe

    def clear(self, *getters):
        for get in getters:
            attr = get.__name__
            setattr(self, attr, "")

    def remove_poignancy_agent(self) -> None:
        self.poignancy = re.sub(r" for .*?\.", ".", self.poignancy)

    def change_response_range(self, lo: int, hi: int) -> None:
        rnge = self._some(re.search(r"\d+ to \d+", self.poignancy)).group(0)
        prev_lo, prev_hi = rnge.split(" ")[0::2]
        self.poignancy = re.sub(
            rf"(\D){prev_hi}(\D)",
            rf"\g<1>{hi}\2",
            re.sub(rf"(\D){prev_lo}(\D)", rf"\g<1>{lo}\2", self.poignancy),
        )
        self.output_fmt = re.sub(
            rf"(\D){prev_hi}(\D)",
            rf"\g<1>{hi}\2",
            re.sub(rf"(\D){prev_lo}(\D)", rf"\g<1>{lo}\2", self.output_fmt),
        )

    def change_output_eg(self, value: int) -> None:
        self.output_fmt = re.sub(r"(\"output\": \")\d", rf"\g<1>{value}", self.output_fmt)

    def as_str(self) -> str:
        return "".join((self.agent, self.poignancy, self.event, self.output_fmt))

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.agent, self.poignancy, self.event, self.output_fmt)

    def as_dict(self) -> dict[str, str]:
        return {
            "agent": self.agent,
            "poignancy": self.poignancy,
            "event": self.event,
            "output_fmt": self.output_fmt,
        }

    def as_series(self) -> pd.Series:
        return pd.Series(self.as_dict())


def reduce_to_unique(prompts: pd.DataFrame) -> dict[str, set[str]]:
    unique: dict[str, set[str]] = {}
    for col, col_data in prompts.iteritems():
        unique[col] = set(col_data)
        print(f"Unique in {col}:", len(unique[col]))
    return unique


# TODO: Add a function which creates all unique combinations of elements of sets of
#       a dict[str, set[str]]


def edit_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    # prompts.map(lambda p: Prompt(p).remove_poignancy_agent())
    # prompts.map(lambda p: Prompt(p).clear(Prompt.agent))
    # prompts.map(lambda p: Prompt(p).change_response_range(1, 7))
    prompts.map(lambda p: Prompt(p).change_output_eg(3))
    return prompts


def main():
    prompts = pd.read_csv(PROMPT_FILE)
    prompts_objs = pd.DataFrame(prompts["prompt"].apply(Prompt))
    prompts_objs = edit_prompts(prompts_objs)
    prompts_new = prompts_objs["prompt"].apply(lambda p: p.as_str())
    print(prompts_new)
    prompts_new.to_csv("smallville_417_maria/prompts_testing.csv", index=False)


if __name__ == "__main__":
    main()

