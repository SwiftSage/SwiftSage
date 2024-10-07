import argparse
import datetime
import json
import logging
import multiprocessing
import os
from time import sleep

from tqdm import tqdm

from swiftsage.agents import SwiftSage
from swiftsage.benchmark.data_loader import load_data
from swiftsage.utils.commons import api_configs, setup_logging
from swiftsage.benchmark.data_utils import parse_question, parse_ground_truth, extract_answer
from swiftsage.benchmark.evaluate import evaluate


logger = setup_logging()


def run_benchmark(swiftsage, args, max_iterations=5, reward_threshold=8):
    examples = load_data(args.dataset_name, args.split, args.data_dir, args.num_test_sample)

    res = []
    skip_ids = []

    output_path = os.path.join(args.output_path, f"{args.dataset_name}.jsonl")
    if os.path.exists(output_path):
        with open(output_path) as fr:
            model_responses = fr.readlines()

        for item in model_responses:
            item = json.loads(item)
            res.append(item)
            skip_ids.append(item["idx"])

    for example in tqdm(examples, desc=args.dataset_name):
        if example["idx"] in skip_ids:
            continue
        question = parse_question(example, args.dataset_name)
        gt_ans = parse_ground_truth(example, args.dataset_name)
        reasoning, raw_solution, messages = swiftsage.solve(question, max_iterations, reward_threshold)
        
        if raw_solution == "No current solution yet.":
            solution = raw_solution
        else:
            solution = raw_solution[len("Answer (from running the code):\n "):]

        cur_res = {
            "idx": example["idx"],
            "question": question,
            "gt": gt_ans,
            "pred": solution,
            "raw_pred": raw_solution,
            "reasoning": reasoning,
        }
        res.append(cur_res)

        with open(output_path, "a") as fw:
            fw.write(json.dumps(res[-1]) + "\n")

        sleep(30)
    
    # Evaluate the results
    res, result_metric = evaluate(res)
    with open(os.path.join(args.output_path, f"{args.dataset_name}_score.jsonl"), "w") as fw:
        for item in res:
            fw.write(json.dumps(item) + "\n")
    with open(os.path.join(args.output_path, f"{args.dataset_name}_metric.jsonl"), "w") as fw:
        fw.write(json.dumps(result_metric) + "\n")


def main(args):
    swift_config = {
        "model_id": args.swift_model_id,
        "api_config": api_configs[args.api_provider]
    }

    reward_config = {
        "model_id": args.feedback_model_id,
        "api_config": api_configs[args.api_provider]
    }

    sage_config = {
        "model_id": args.sage_model_id,
        "api_config": api_configs[args.api_provider]
    }

    # specify the path to the prompt templates
    prompt_template_dir = args.prompt_template_dir
    dataset = [] 
    embeddings = [] # TODO: for retrieval augmentation (not implemented yet now)
    s2 = SwiftSage(
        dataset,
        embeddings,
        prompt_template_dir,
        swift_config,
        sage_config,
        reward_config,
        use_retrieval=args.use_retrieval,
        start_with_sage=args.start_with_sage,
    )

    run_benchmark(s2, args, args.max_iterations, args.reward_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="MATH", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data

    parser.add_argument("--api_provider", default="Together", choices=["Together", "SambaNova"], type=str)
    parser.add_argument("--swift_model_id", default="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", type=str)
    parser.add_argument("--feedback_model_id", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", type=str)
    parser.add_argument("--sage_model_id", default="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", type=str)

    parser.add_argument("--prompt_template_dir", default='./swiftsage/prompt_templates', type=str)
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--start_with_sage", action="store_true")

    parser.add_argument("--max_iterations", default=5, type=int)
    parser.add_argument("--reward_threshold", default=8, type=int)

    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.api_provider == "SambaNova":
        args.swift_model_id = args.swift_model_id.split("/")[-1][:-len("Turbo")]
        args.feedback_model_id = args.feedback_model_id.split("/")[-1][:-len("Turbo")]
        args.sage_model_id = args.sage_model_id.split("/")[-1][:-len("Turbo")]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    multiprocessing.set_start_method('spawn')
    main(args)
