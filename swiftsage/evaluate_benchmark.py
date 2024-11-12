import argparse
import json
import logging
import multiprocessing
import os

from tqdm import tqdm

from swiftsage.agents import SwiftSage
from swiftsage.embedding_models import JinaModel, JinaAPIModel
from swiftsage.benchmark.data_loader import load_data
from swiftsage.utils import get_index
from swiftsage.utils.commons import api_configs
from swiftsage.benchmark.evaluate import evaluate_math, evaluate_multiple_choice


logger = logging.getLogger("SwiftSage")


def run_benchmark(swiftsage, args, max_iterations=3, reward_threshold=1):
    examples = load_data(args.dataset_name)

    # decide start_index and end_index by num_shards and shard_id  
    if args.num_shards > 1:
        examples = examples.shard(num_shards=args.num_shards, index=args.shard_id, contiguous=True)

    res = []
    skip_ids = []

    output_path = os.path.join(args.output_path, f"{args.dataset_name}.jsonl")
    if os.path.exists(output_path):
        with open(output_path) as fr:
            model_responses = fr.readlines()

        for item in model_responses:
            item = json.loads(item)
            res.append(item)
            skip_ids.append(item["id"])

    for example in tqdm(examples, desc=args.dataset_name):
        if example["id"] in skip_ids:
            continue
        question = example["question"]
        gt_ans = example["answer"]
        reasoning, solution, messages, raw_response = swiftsage.solve(question, max_iterations, reward_threshold)

        cur_res = {
            "id": example["id"],
            "question": question,
            "gt": gt_ans,
            "pred": solution,
            "reasoning": reasoning,
        }
        res.append(cur_res)

        with open(output_path, "a") as fw:
            fw.write(json.dumps(res[-1]) + "\n")

        json.dump({"messages": messages, "raw_response": raw_response}, open(os.path.join(args.output_path, "logs", f"{example['id']}.json"), "w"), indent=2)
    
    # Evaluate the results
    if args.dataset_name in ["math-l5"]:
        res, result_metric = evaluate_math(res)
    elif args.dataset_name in ["gpqa"]:
        res, result_metric = evaluate_multiple_choice(res)
    with open(os.path.join(args.output_path, f"{args.dataset_name}_score.jsonl"), "w") as fw:
        for item in res:
            fw.write(json.dumps(item) + "\n")
    with open(os.path.join(args.output_path, f"{args.dataset_name}_metric.jsonl"), "w") as fw:
        fw.write(json.dumps(result_metric) + "\n")


def main(args):
    swift_config = {
        "model_id": args.swift_model_id,
        "api_config": api_configs[args.swift_api_provider]
    }

    reward_config = {
        "model_id": args.feedback_model_id,
        "api_config": api_configs[args.feedback_api_provider]
    }

    sage_config = {
        "model_id": args.sage_model_id,
        "api_config": api_configs[args.sage_api_provider]
    }

    # specify the path to the prompt templates
    prompt_template_dir = args.prompt_template_dir

    embedding_model, retrieval_dataset, embeddings = None, None, None
    if args.use_retrieval:
        if args.embedding_model_type == "jina_api":
            embedding_model = JinaAPIModel(
                args.embedding_model_name,
            )
        elif args.embedding_model_type == "jina":
            embedding_model = JinaModel(
                args.embedding_model_name,
            )
        else:
            raise ValueError("Invalid embedding model type")

        retrieval_dataset = json.load(open(os.path.join(args.retrieval_dataset_path)))
        embeddings = get_index(embedding_model, retrieval_dataset, args.embedding_dim, args.index_path)

    multiple_choice = args.dataset_name in ["gpqa"]
    if multiple_choice:
        multiple_choice_config = {
            "model_id": args.swift_model_id,
            "api_config": api_configs[args.swift_api_provider]
        }
    else:
        multiple_choice_config = None

    s2 = SwiftSage(
        prompt_template_dir=prompt_template_dir,
        swift_config=swift_config,
        sage_config=sage_config,
        feedback_config=reward_config,
        multiple_choice=multiple_choice,
        multiple_choice_config=multiple_choice_config,
        only_swift=args.only_swift,
        best_of_n=args.best_of_n,
        retrieval_dataset=retrieval_dataset,
        embeddings=embeddings,
        embedding_model=embedding_model,
        use_retrieval=args.use_retrieval,
    )

    run_benchmark(s2, args, args.max_iterations, args.reward_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="MATH", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)

    parser.add_argument("--api_provider", default="Together", choices=list(api_configs.keys()), type=str)
    parser.add_argument("--swift_api_provider", choices=list(api_configs.keys()), type=str)
    parser.add_argument("--feedback_api_provider", choices=list(api_configs.keys()), type=str)
    parser.add_argument("--sage_api_provider", choices=list(api_configs.keys()), type=str)
    parser.add_argument("--swift_model_id", default="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", type=str)
    parser.add_argument("--feedback_model_id", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", type=str)
    parser.add_argument("--sage_model_id", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", type=str)

    parser.add_argument("--prompt_template_dir", default='./swiftsage/prompt_templates', type=str)
    parser.add_argument("--only_swift", action="store_true")
    parser.add_argument("--best_of_n", default=1, type=int)

    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--embedding_model_type", choices=["jina_api", "jina"], type=str)
    parser.add_argument("--embedding_model_name", default="jina-embeddings-v3", type=str)
    parser.add_argument("--embedding_dim", default=1024, type=int)
    parser.add_argument("--index_path", default="./index", type=str)
    parser.add_argument("--retrieval_dataset_path", default="./data/retrieval", type=str)

    parser.add_argument("--max_iterations", default=3, type=int)
    parser.add_argument("--reward_threshold", default=1, type=int)

    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--overwrite", action="store_true")
    
    parser.add_argument("--run_name", default="", type=str)

    args = parser.parse_args()

    args.swift_api_provider = args.swift_api_provider or args.api_provider
    args.feedback_api_provider = args.feedback_api_provider or args.api_provider
    args.sage_api_provider = args.sage_api_provider or args.api_provider

    if args.run_name:
        args.output_path = os.path.join(args.output_path, args.dataset_name, args.run_name)
    else:
        args.output_path = os.path.join(args.output_path, args.dataset_name, f"{args.swift_model_id.split('/')[-1]}")
    
    if not os.path.exists(args.output_path):
        os.system(f"mkdir -p {args.output_path}")
        os.system(f"mkdir -p {args.output_path}/logs")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    multiprocessing.set_start_method('spawn')
    main(args)
