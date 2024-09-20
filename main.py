import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod
import openai
import json 
import logging
import datetime
import re
import hjson

from utils import extract_and_parse_markup, Agent, api_configs, LLMClient, PromptTemplate, logger

class RetrievalAugmentation:
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
        self.reasoning_time = {}
        self.solution_time = {}

    def generate_response(self, prompt, reasoning, current_solution, plan, critical_feedback):
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
            {"role": "user", "content": swift_prompt},
            # {"role": "assistant", "content": "\n<reasoning_steps>"} # prefix-filling 
        ]
        
        response = self.llm_client.generate_response(messages) 
        
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
        

    def generate_response(self, prompt, reasoning, current_solution):
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
        
        response = self.llm_client.generate_response(messages)
        # logger.info(f"SageAgent raw response:\n{response}")
        
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

    def calculate_reward(self, problem, reasoning, current_solution):
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
        
        reward_response = self.llm_client.generate_response(messages) 
        
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
    def __init__(self, dataset, embeddings, prompt_template_dir, swift_config, sage_config, reward_config, use_retrieval=True, start_with_sage=True):
        prompt_template = PromptTemplate(prompt_template_dir)
        retrieval_augmentation = RetrievalAugmentation(dataset, embeddings) if use_retrieval else None
        
        swift_llm = LLMClient(**swift_config)
        sage_llm = LLMClient(**sage_config)
        reward_llm = LLMClient(**reward_config)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.reward_model = RewardModel(prompt_template, reward_llm)
        self.start_with_sage = start_with_sage
    
    def solve(self, problem, max_iterations=10, reward_threshold=8):
        logger.info(f"Starting to solve problem: {problem}")
        current_solution = "No current solution yet." # final answer
        current_reasoning = "No reasoning steps yet." # reasoning steps
        plan = "Initial plan: Take a deep breath and think step by step."
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback
        solved = False
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}")

            if (i == 0 and self.start_with_sage) or self.reward_model.should_consult_sage():
                sage_parsed = self.sage.generate_response(problem, current_reasoning, current_solution) 
                critical_feedback = sage_parsed["critical_feedback"]
                # plan = "\n - " + "\n - ".join(sage_parsed["revised_plan"])
                reasoning = sage_parsed["reasoning_steps"]
                current_reasoning = reasoning    
                solved = sage_parsed["solved"].lower() == "true" if i != 0 else sage_parsed["solved"] 
                logger.info(f"Sage's feedback (iteration {i+1}):\n{critical_feedback}")
                # logger.info(f"Sage's revised plan:\n{plan}")
                logger.info(f"Sage's reasoning steps:\n{current_reasoning}")
                self.sage.feedbacks[i] = critical_feedback
                # self.sage.plans[i] = plan
                current_solution = sage_parsed["final_answer"]
                logger.info("Activated Sage, so we should return the reasoning and solution from Sage.")
                return reasoning, current_solution
            
            if not solved:
                swift_parsed = self.swift.generate_response(problem, current_reasoning, current_solution, plan, critical_feedback)
                
                if "final_answer" not in swift_parsed:
                    logger.info("Swift's response does not contain the 'final_answer' field. Returning raw response.")
                    self.reward_model.scores.append(0)
                    self.reward_model.feedbacks.append("No feedback")
                    self.reward_model.stagnant_count += 5
                    continue 
                
                current_solution = swift_parsed["final_answer"]
                reasoning = swift_parsed.get("reasoning_steps", json.dumps(swift_parsed))

                self.swift.reasoning_time[i] = reasoning
                self.swift.solution_time[i] = current_solution

                logger.info(f"Swift's reasoning:\n{reasoning}")
                logger.info(f"Swift's current solution:\n{current_solution}")
            
                reward_parsed = self.reward_model.calculate_reward(problem, reasoning, current_solution)
                
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
                    current_solution = self.swift.solution_time[i-1]
                    reasoning = self.swift.reasoning_time[i-1]
                    continue 

                
                critical_feedback = feedback
                current_reasoning = reasoning

            
            if score >= reward_threshold or solved:
                logger.info("Perfect solution found!")
                return reasoning, current_solution 
            
            
            if self.reward_model.should_consult_sage():
                logger.info("Reward model: The solution quality hasn't improved recently. Consulting Sage for the next iteration.")
        
        logger.info("Max iterations reached without finding a perfect solution.")
        logger.info("Problem solving completed")
        return reasoning, current_solution

# for retrieval augmentation (not implemented yet now)
dataset = ["Example problem 1: ...", "Example problem 2: ...", "Example problem 3: ..."]
embeddings = np.random.rand(len(dataset), 768)  # Placeholder, replace with actual embeddings

# specify the path to the prompt templates
prompt_template_dir = "./prompt_templates"

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
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "api_config": api_configs['Together']
}

reward_config = {
    "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "api_config": api_configs['Together']
}

sage_config = {
    "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "api_config": api_configs['Together']
}


s2 = SwiftSage(
    dataset, 
    embeddings, 
    prompt_template_dir, 
    swift_config,
    sage_config,
    reward_config,
    use_retrieval=False,
    start_with_sage=False
)

# problem = "Solve the equation: 2x + 5 = 13"
# problem = "If h(x)=x-4 and g(h(x))=x^2-8x+10, find g(x)? show the formula for g(x)"
# problem = "Solve the equation: 6y + 5 = 29"
# problem = "Who lives longer, Lowell Sherman or Jonathan Kaplan?"
# problem = "9.9 or 9.11 --  which is bigger?"
# problem = "How can you solve the quadratic equation 3x^2 + 7x + 4 = 0 using the quadratic formula?"
# problem = "Explain why sound waves cannot travel in a vacuum?"
# problem = "How many grams of hydrogen (H) are present in 23.5 grams of water (H2O)?"
# problem = "What is the distance between the points (2, 3) and (5, 8)?"
# problem = "Why can the Hubble telescope capture clear images of distant stars and galaxies, but not a detailed image of Pluto?"
# problem = """
# A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?
# """

# problem = "Tim wants to invest some money in a bank which compounds quarterly with an annual interest rate of $7\%$. To the nearest dollar, how much money should he invest if he wants a total of $\$60,\!000$ at the end of $5$ years?"

# problem = """
# In an SR latch built from NOR gates, which condition is not allowed

# Options:
# [ "S=0, R=2", "S=2, R=2", "S=1, R=1", "S=1, R=-1", "S=1, R=2", "S=0, R=0", "S=2, R=0", "S=1, R=0", "S=2, R=1", "S=0, R=1" ]

# Which one is the correct answer?

# """

# https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/viewer/default/validation?q=Let+A+be+the+set+of+all+&row=1https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/viewer/default/validation?q=Let+A+be+the+set+of+all+&row=1
problem = """
Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?

Options:
[ "ST + TS is the identity map of V onto itself.", "TS = 0", "ST = 1", "ST - TS = 0", "ST = T", "ST = 0", "ST = TS", "ST - TS is the identity map of V onto itself.", "TS = T", "ST = S" ]

Which one is the correct answer?

"""

# GPQA
problem = """
Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they be clearly resolved?

Choices: 
["10^-4 ev", "10^-8 ev", "10^-9 ev", "10^-11 ev"]

Which one is the correct answer?
"""

problem = """
Consider the following metric: ds^{2}=\frac{32}{\left(4-x^{2}-y^{2}\right)}\left(dx^{2}+dy^{2}\right) What is the area of the pseudosphere of radius r=2?

Choices:
["+\infty", "0", "4\pi\left(x^{2}+y^{2}\right)", "4\pi\left(x^{2}-y^{2}\right)"]

Which one is the correct answer?
"""

# AIME 

problem = """
Two externally tangent circles $\omega_1$ and $\omega_2$ have centers $O_1$ and $O_2$, respectively. A third circle $\Omega$ passing through $O_1$ and $O_2$ intersects $\omega_1$ at $B$ and $C$ and $\omega_2$ at $A$ and $D$, as shown. Suppose that $AB = 2$, $O_1O_2 = 15$, $CD = 16$, and $ABO_1CDO_2$ is a convex hexagon. Find the area of this hexagon.
"""


reasoning, solution = s2.solve(problem, max_iterations=5, reward_threshold=8)
logger.info(f"Final reasoning:\n{reasoning}")
logger.info(f"Final solution:\n{solution}")

