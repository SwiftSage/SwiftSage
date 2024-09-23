import argparse
import datetime
import json
import logging
import multiprocessing
import os
import re
from abc import ABC, abstractmethod

import hjson
import numpy as np
import openai
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_data
from code_executor import PythonExecutor
from utils import (Agent, LLMClient, PromptTemplate, api_configs,
                   extract_and_parse_markup, setup_logging)
from data_utils import parse_question, parse_ground_truth
from evaluate import evaluate


logger = setup_logging()

class RetrievalAugmentation:
    # TODO: implement the retrieval augmentation later 
    def __init__(self, dataset, embeddings):
        self.dataset = dataset
        self.embeddings = embeddings

    def get_similar_examples(self, query_embedding, n=3):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]
        return [self.dataset[i] for i in top_indices]

class SwiftAgent(Agent):
    def __init__(self, prompt_template, llm_client, retrieval_augmentation=None):
        super().__init__(prompt_template, llm_client)
        self.retrieval_augmentation = retrieval_augmentation
        self.plans = {}
        self.codes = {}

    def generate_response(self, prompt, reasoning, current_solution, plan, critical_feedback, prefill=True):
        logger.info("SwiftAgent generating response")
        if self.retrieval_augmentation:
            query_embedding = self.get_query_embedding(prompt)
            similar_examples = self.retrieval_augmentation.get_similar_examples(query_embedding)
            examples_text = "\n".join(similar_examples) # TODO: add more context to the prompt
        else:
            examples_text = "No similar examples available."
        
        swift_prompt = self.prompt_template.format(
            "swift",
            prompt=prompt,
            current_reasoning=reasoning, # TODO: check if this is needed
            examples=examples_text,
            current_solution=current_solution,
            critical_feedback=critical_feedback,
            revised_plan=plan
        )
        # logger.info(f"SwiftAgent prompt:\n{swift_prompt}")

        messages = [
            {"role": "system", "content": ''},
            {"role": "user", "content": swift_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<plan>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages) 
        if prefill:
            response = "<plan>" + response
        
        try:
            parsed_response = extract_and_parse_markup(response)
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Swift's response was not in valid JSON format. Returning raw response.")
            return response

    def get_query_embedding(self, query):
        # Implement query embedding generation
        return np.random.rand(768)  # Placeholder, replace with actual embedding
 
class SageAgent(Agent):
    def __init__(self, prompt_template, llm_client):
        super().__init__(prompt_template, llm_client)
        self.feedbacks = {}
        self.plans = {}
        

    def generate_response(self, prompt, reasoning, current_solution, prefill=True):
        logger.info("SageAgent generating response")
        sage_prompt = self.prompt_template.format(
            "sage",
            prompt=prompt,
            reasoning=reasoning, 
            current_solution=current_solution
        )
        # logger.info(f"SageAgent prompt:\n{sage_prompt}")
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": sage_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<solved>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages)
        # logger.info(f"SageAgent raw response:\n{response}")
        if prefill:
            response = "<solved>" + response
        try:
            parsed_response = extract_and_parse_markup(response)
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Sage's response was not in valid JSON format. Returning raw response.")
            return response

class RewardModel:
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client 
        self.scores = [] 
        self.feedbacks = [] 
        self.stagnant_count = 0

    def calculate_reward(self, problem, reasoning, current_solution, prefill=True):
        reward_prompt = self.prompt_template.format(
            "reward",
            problem=problem,
            reasoning= reasoning,
            current_solution=current_solution
        )
        # logger.info(f"RewardModel prompt:\n{reward_prompt}")
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": reward_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<feedback>"}) # prefix-filling 
        
        reward_response = self.llm_client.generate_response(messages) 
        if prefill:
            reward_response = "<feedback>" + reward_response
        
        try:
            parsed_response = extract_and_parse_markup(reward_response)
            score = int(parsed_response["score"])
            
            # Update stagnant_count based on score comparison
            if len(self.scores) > 0 and score <= self.scores[-1]:
                self.stagnant_count += 1
            else:
                self.stagnant_count = 0
            
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Reward model's response was not in valid JSON format. Returning raw response.")
            return reward_response

    def should_consult_sage(self):
        # This method remains unchanged
        return self.stagnant_count >= 1 or (len(self.scores) > 0 and self.scores[-1] < 5)

class SwiftSage:
    def __init__(self, dataset, embeddings, prompt_template_dir, swift_config, sage_config, reward_config, use_retrieval=True, start_with_sage=False):
        prompt_template = PromptTemplate(prompt_template_dir)
        retrieval_augmentation = RetrievalAugmentation(dataset, embeddings) if use_retrieval else None
        
        # add logger to the following LLMClient 
        swift_llm = LLMClient(**swift_config, logger=logger)
        sage_llm = LLMClient(**sage_config, logger=logger)
        reward_llm = LLMClient(**reward_config, logger=logger)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.reward_model = RewardModel(prompt_template, reward_llm)
        self.start_with_sage = start_with_sage
        # self.executor = PythonExecutor(get_answer_from_stdout=True)
    
    def solve(self, problem, max_iterations=10, reward_threshold=8):
        logger.info(f"Starting to solve problem: {problem}")
        current_solution = "No current solution yet." # final answer
        current_reasoning = "No reasoning steps yet." # reasoning steps
        plan = "Initial plan: Take a deep breath and think step by step."
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback
        solved = False
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}")
            

            #  Use the Sage Agent 
            if (i == 0 and self.start_with_sage) or self.reward_model.should_consult_sage():
                sage_parsed = self.sage.generate_response(problem, current_reasoning, current_solution) 
                critical_feedback = sage_parsed["critical_feedback"]
                # plan = "\n - " + "\n - ".join(sage_parsed["revised_plan"])
                current_reasoning = sage_parsed["reasoning_steps"] 
                current_code = sage_parsed["code"] 

                solved = sage_parsed["solved"].lower() == "true" if i != 0 else sage_parsed["solved"] 
                if solved:
                    return current_reasoning, current_solution
                logger.info(f"Sage's feedback (iteration {i+1}):\n{critical_feedback}")
                # logger.info(f"Sage's reasoning steps:\n{current_reasoning}")
                self.sage.feedbacks[i] = critical_feedback
                
                # run the code 
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                logger.info(f"Sage Code execution report: {code_report}")
                logger.info(f"Sage Code execution result: {code_result}")
                current_reasoning = current_reasoning + f"\n The generated code is:\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result
                
                # current_solution = sage_parsed["final_answer"]
                logger.info("Activated Sage, so we should return the reasoning and solution from Sage.")
                return current_reasoning, current_solution
            
            if not solved:
                # Use the Swift Agent 
                swift_parsed = self.swift.generate_response(problem, current_reasoning, current_solution, plan, critical_feedback)
                
                if "code" not in swift_parsed and "final_answer" not in swift_parsed: 
                    logger.info("Swift's response does not contain the 'final_answer' or 'code' field. Returning raw response.")
                    self.reward_model.scores.append(0)
                    self.reward_model.feedbacks.append("No feedback")
                    self.reward_model.stagnant_count += max_iterations # force to use Sage Agent
                    continue 
                
                current_plan = swift_parsed["plan"]
                current_code = swift_parsed["code"]
                current_answer = swift_parsed.get("final_answer", None)

                self.swift.plans[i] = current_plan
                self.swift.codes[i] = current_code  

                logger.info(f"Swift's plan:\n{current_plan}")
                logger.info(f"Swift's code:\n{current_code}")

                # Call sandbox to run the code and get the result
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                logger.info(f"Code execution report: {code_report}")
                logger.info(f"Code execution result: {code_result}")
            
                current_reasoning = current_plan + f"\n The generated code is:\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result

                # Calling the reward model to provide feedback and score 
                reward_parsed = self.reward_model.calculate_reward(problem, current_reasoning, current_solution)
                score = int(reward_parsed["score"])
                feedback = reward_parsed["feedback"] 
                prev_score = self.reward_model.scores[-1] if len(self.reward_model.scores) > 0 else 0
                self.reward_model.scores.append(score)
                self.reward_model.feedbacks.append(feedback)

                # detect if the score is lower than the previous score
                logger.info(f"Reward for iteration {i+1}: {score}/10")
                logger.info(f"Feedback: {feedback}")

                if False and score < prev_score:
                    logger.info("Score is lower than the previous score. Stopping the iteration. Reverting to the previous solution and reasoning.")
                    # revert to the previous solution and reasoning
                    current_solution = self.swift.codes[i-1]
                    current_reasoning = self.swift.plans[i-1]
                    continue 

                
                critical_feedback = feedback 

            
            if score >= reward_threshold or solved:
                logger.info("Perfect solution found!")
                return current_reasoning, current_solution 
            
            
            if self.reward_model.should_consult_sage():
                logger.info("Reward model: The solution quality hasn't improved recently. Consulting Sage for the next iteration.")
        
        logger.info("Max iterations reached without finding a perfect solution.")
        logger.info("Problem solving completed")
        return current_reasoning, current_solution


def run_test(swiftsage, problem, max_iterations=5, reward_threshold=8):
    logger.info(f"Testing problem: {problem}")
    reasoning, solution = swiftsage.solve(problem, max_iterations, reward_threshold)
    logger.info(f"Final reasoning:\n{reasoning}")
    logger.info(f"Final solution:\n{solution}")
    logger.info("=" * 50)


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
        reasoning, solution = swiftsage.solve(question, max_iterations, reward_threshold)
        
        # TODO: extract answer from solution

        cur_res = {
            "idx": example["idx"],
            "question": question,
            "gt": gt_ans,
            "pred": solution,
            "reasoning": reasoning,
        }
        res.append(cur_res)

        with open(output_path, "a") as fw:
            fw.write(json.dumps(res[-1]) + "\n")
    
    # Evaluate the results
    res, result_metric = evaluate(res)
    with open(args.output_path, f"{args.dataset_name}_score.jsonl", "w") as fw:
        for item in res:
            fw.write(json.dumps(item) + "\n")
    with open(args.output_path, f"{args.dataset_name}_metric.jsonl", "w") as fw:
        fw.write(json.dumps(result_metric) + "\n")


def main(args):

    # TODO: for retrieval augmentation (not implemented yet now) 
    # dataset = ["Example problem 1: ...", "Example problem 2: ...", "Example problem 3: ..."]
    # embeddings = np.random.rand(len(dataset), 768)  # Placeholder, replace with actual embeddings


    # Configuration for each LLM
    # swift_config = {
    #     "model_id": "Meta-Llama-3.1-8B-Instruct",
    #     "api_config": api_configs['SambaNova']
    # }

    # reward_config = {
    #     "model_id": "Meta-Llama-3.1-70B-Instruct",
    #     "api_config": api_configs['SambaNova']
    # }

    # sage_config = {
    #     "model_id": "Meta-Llama-3.1-405B-Instruct",
    #     "api_config": api_configs['SambaNova']
    # }

    swift_config = {
        "model_id": args.swift_model_id,
        "api_config": api_configs[args.api_provider]
    }

    reward_config = {
        "model_id": args.reward_model_id,
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

    if args.eval_mode == "test":
        test_problems = [
            "Solve the equation: 2x + 5 = 13", # 0
            "If h(x)=x-4 and g(h(x))=x^2-8x+10, find g(x)? show the formula for g(x)", # 1
            "Solve the equation: 6y + 5 = 29", # 2
            "Who lives longer, Lowell Sherman or Jonathan Kaplan?", # 3
            "9.9 or 9.11 --  which is bigger?", # 4
            "How can you solve the quadratic equation 3x^2 + 7x + 4 = 0 using the quadratic formula?", # 5
            "Explain why sound waves cannot travel in a vacuum?", # 6
            "How many grams of hydrogen (H) are present in 23.5 grams of water (H2O)?", # 7
            "What is the distance between the points (2, 3) and (5, 8)?", # 8
            "Why can the Hubble telescope capture clear images of distant stars and galaxies, but not a detailed image of Pluto?", # 9
            """A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?""",
            """Tim wants to invest some money in a bank which compounds quarterly with an annual interest rate of $7\%$. To the nearest dollar, how much money should he invest if he wants a total of $\$60,\!000$ at the end of $5$ years?""",
            """In an SR latch built from NOR gates, which condition is not allowed

            Options:
            [ "S=0, R=2", "S=2, R=2", "S=1, R=1", "S=1, R=-1", "S=1, R=2", "S=0, R=0", "S=2, R=0", "S=1, R=0", "S=2, R=1", "S=0, R=1" ]

            Which one is the correct answer?""",
            # ... add other problems here ...
            """How many letter r are there in the word "strawberry"?"""
        ]

        # for problem in test_problems:
        pid = 7
        print(f"Problem {pid}: {test_problems[pid]}")
        run_test(s2, test_problems[pid], args.max_iterations, args.reward_threshold)
    elif args.eval_mode == "benchmark":
        run_benchmark(s2, args, args.max_iterations, args.reward_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", default="test", choices=["test", "benchmark"], type=str)

    parser.add_argument("--dataset_name", default="MATH", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data

    parser.add_argument("--api_provider", default="Together", choices=["Together", "SambaNova"], type=str)
    parser.add_argument("--swift_model_id", default="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", type=str)
    parser.add_argument("--reward_model_id", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", type=str)
    parser.add_argument("--sage_model_id", default="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", type=str)

    parser.add_argument("--prompt_template_dir", default='./prompt_templates', type=str)
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--start_with_sage", action="store_true")

    parser.add_argument("--max_iterations", default=5, type=int)
    parser.add_argument("--reward_threshold", default=8, type=int)

    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # remove console output for benchmark evaluation
    if args.eval_mode != "test":
        root_logger = logging.getLogger("")
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
                break

    if args.api_provider == "SambaNova":
        args.swift_model_id = args.swift_model_id.split("/")[-1][:-len("Turbo")]
        args.reward_model_id = args.reward_model_id.split("/")[-1][:-len("Turbo")]
        args.sage_model_id = args.sage_model_id.split("/")[-1][:-len("Turbo")]

    multiprocessing.set_start_method('spawn')
    main(args)
