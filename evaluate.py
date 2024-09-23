"""
Source and credits: https://github.com/ZubinGou/math-evaluation-harness/blob/main/python_executor.py
"""
import argparse
import json
from concurrent.futures import TimeoutError

import numpy as np
from pebble import ProcessPool
from tqdm import tqdm

from grader import math_equal_process


def evaluate(samples: list=None, file_path: str=None):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]

    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    params = [(idx, sample['pred'], sample['gt']) for idx, sample in enumerate(samples)]

    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    assert len(samples) == len(scores)

    for i in range(len(samples)):
        samples[i]['score'] = scores[i]

    mean_score = np.round(np.mean([score for score in scores if score is not False]), decimals=2)

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "acc": mean_score
    }

    return samples, result_json


if __name__ == "__main__":
    samples, results_json = evaluate(file_path="output/MATH.jsonl")
    print('test')
