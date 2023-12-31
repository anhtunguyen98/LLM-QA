import json
from dataclasses import dataclass, field
from typing import Optional

from datasets import DatasetDict, load_dataset
from transformers import HfArgumentParser

from utils import DEFAULT_SYSTEM_PROMPT, get_prompt


@dataclass
class ScriptArguments:
    prompt: Optional[str] = field(
        default="single_turn",
        metadata={"help": "single_turn, multi_turn"},
    )
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def get_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = ["?"]
    answers = json.dumps(answers) if all_answers else f'"{answers[0]}"'

    return {
        "text": get_prompt(
            f"""\
Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```
If the answer is not in the context, the answer should be "?".
Context: {context}
Question: {question}""",
            [],
            SYSTEM_PROMPT,
        )
        + f""" \
```json
{{
  "answer": {answers}
}}
``` </s>"""
    }


instruction = get_prompt_and_response



squad_dataset = load_dataset("squad_v2")
train_dataset = squad_dataset["train"].map(instruction)
print(train_dataset[0]["text"])
test_dataset = squad_dataset["validation"].map(
    instruction, fn_kwargs={"all_answers": True}
)
print(test_dataset[0]["text"])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset.save_to_disk(script_args.dataset)