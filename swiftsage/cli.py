import argparse
import multiprocessing
import random

from swiftsage.agents import SwiftSage
from swiftsage.utils.commons import api_configs, setup_logging
from pkg_resources import resource_filename


logger = setup_logging()


def run_test(swiftsage, problem, max_iterations=5, reward_threshold=8):
    logger.info(f"Testing problem: {problem}")
    reasoning, solution = swiftsage.solve(problem, max_iterations, reward_threshold)
    logger.info(f"Final reasoning:\n{reasoning}")
    logger.info(f"Final solution:\n{solution}")
    logger.info("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--problem", type=str)

    parser.add_argument("--api_provider", default="Together", choices=["Together", "SambaNova"], type=str)
    parser.add_argument("--swift_model_id", default="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", type=str)
    parser.add_argument("--feedback_model_id", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", type=str)
    parser.add_argument("--sage_model_id", default="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", type=str)
    
    default_template_dir = resource_filename('swiftsage', 'prompt_templates')
    parser.add_argument("--prompt_template_dir", default=default_template_dir, type=str)
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--start_with_sage", action="store_true")

    parser.add_argument("--max_iterations", default=5, type=int)
    parser.add_argument("--reward_threshold", default=8, type=int)

    args = parser.parse_args()

    if args.api_provider == "SambaNova":
        args.swift_model_id = args.swift_model_id.split("/")[-1][:-len("Turbo")]
        args.feedback_model_id = args.feedback_model_id.split("/")[-1][:-len("Turbo")]
        args.sage_model_id = args.sage_model_id.split("/")[-1][:-len("Turbo")]

    return args


def main():
    args = parse_args()
    multiprocessing.set_start_method('spawn')
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

    if not args.problem:
        problem = random.choice(test_problems)
        print(f"Problem: {problem}")
    else:
        problem = args.problem
    run_test(s2, problem, args.max_iterations, args.reward_threshold)


if __name__ == '__main__':
    main()
