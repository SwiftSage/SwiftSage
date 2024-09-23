import json
import os
import re
import random
from typing import Any, Iterable, Union

from datasets import Dataset, concatenate_datasets, load_dataset

from data_utils import (
    lower_keys,
    parse_question,
    parse_ground_truth,
)


def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def load_data(
        data_name, 
        split='test', 
        data_dir='./data',
        num_test_sample=-1,
    ):
    if data_name.lower() == "math":
        data_name = 'MATH'  # we use 500 problem test split in "Let's Verify Step-by-Step"
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", 'all', split='test')
            # only keep stem subjects
            stem_subjects = ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
                'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
                'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology',
                'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics',
                'high_school_physics', 'high_school_statistics', 'machine_learning']
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x['type'] in stem_subjects)
        elif data_name == "mathvista":
            raise NotImplementedError(data_name)
        elif data_name == "gpqa":
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        elif data_name == "codeforces":
            raise NotImplementedError(data_name)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if 'idx' not in examples[0]:
        examples = [{'idx': i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x['idx'])

    if num_test_sample > 0:
        examples = examples[:num_test_sample]

    return examples


if __name__ == "__main__":
    examples = load_data("gpqa", "test")
    print('test')
