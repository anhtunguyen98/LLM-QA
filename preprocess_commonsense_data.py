import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from datasets import DatasetDict, load_dataset
import datasets
from transformers import HfArgumentParser
import pandas as pd
from utils import DEFAULT_SYSTEM_PROMPT, get_prompt


@dataclass
class ScriptArguments:
    dataset: Optional[str] = field(
        default="data/commonsense_qa",
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def get_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answerKey"]
    choices = ''
    for label,text in zip(item['choices']['label'],item['choices']['text']):
        choices += f' {label}. {text}'
    if len(answers) == 0:
        answers = ["?"]
    answers = json.dumps(answers) if all_answers else f'"{answers[0]}"'

    return {
        "text": get_prompt(
            f"""\
You are given a question, context and a set of choices. Your task is to select the correct choice that best answers the question (A, B, C, D or E). Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```.
Context: {context}
Question: {question}
Choices:{choices}""",
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

def add_context(dataset,file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
    samples = []
    for i,x in enumerate(dataset):
        id = x['id']
        x['context'] = data[id]
        samples.append(x)
    df = pd.DataFrame(samples)
    return datasets.Dataset.from_pandas(df)

instruction =  get_prompt_and_response

squad_dataset = load_dataset("commonsense_qa")
train_dataset = add_context(squad_dataset["train"],'data/train.json')
train_dataset = train_dataset.map(instruction)
test_dataset = add_context(squad_dataset["validation"],'data/valid.json')
test_dataset = test_dataset.map(instruction)


print(test_dataset[10]['text'])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset.save_to_disk(script_args.dataset)