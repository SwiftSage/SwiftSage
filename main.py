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

from utils import extract_and_parse_json, Agent, api_configs, LLMClient, PromptTemplate, logger

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

    def generate_response(self, prompt, reasoning, current_solution, plan, critical_feedback):
        logger.info("SwiftAgent generating response")
        if self.retrieval_augmentation:
            query_embedding = self.get_query_embedding(prompt)
            similar_examples = self.retrieval_augmentation.get_similar_examples(query_embedding)
            examples_text = "\n".join(similar_examples)
        else:
            examples_text = "No similar examples available."
        
        swift_prompt = self.prompt_template.format(
            "swift",
            prompt=prompt,
            reasoning=reasoning,
            examples=examples_text,
            current_solution=current_solution,
            critical_feedback=critical_feedback,
            revised_plan=plan
        )
        logger.info(f"SwiftAgent prompt:\n{swift_prompt}")

        messages = [
            {"role": "system", "content": "You are a problem-solving agent."},
            {"role": "user", "content": swift_prompt}
        ]
        
        response = self.llm_client.generate_response(messages) 
        
        try:
            parsed_response = extract_and_parse_json(response)
            return json.dumps(parsed_response, indent=2)
        except json.JSONDecodeError:
            logger.error("Error: Swift's response was not in valid JSON format. Returning raw response.")
            return response

    def get_query_embedding(self, query):
        # Implement query embedding generation
        return np.random.rand(768)  # Placeholder, replace with actual embedding
 
class SageAgent(Agent):
    def generate_response(self, prompt, reasoning, current_solution):
        logger.info("SageAgent generating response")
        sage_prompt = self.prompt_template.format(
            "sage",
            prompt=prompt,
            reasoning=reasoning, 
            current_solution=current_solution
        )
        logger.info(f"SageAgent prompt:\n{sage_prompt}")
        
        messages = [
            {"role": "system", "content": "You are a high-level problem-solving agent."},
            {"role": "user", "content": sage_prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        logger.info(f"SageAgent raw response:\n{response}")
        
        try:
            parsed_response = extract_and_parse_json(response)
            return json.dumps(parsed_response, indent=2)
        except json.JSONDecodeError:
            logger.error("Error: Sage's response was not in valid JSON format. Returning raw response.")
            return response

class RewardModel:
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client
        self.previous_score = 0
        self.stagnant_count = 0

    def calculate_reward(self, problem, reasoning, current_solution):
        reward_prompt = self.prompt_template.format(
            "reward",
            problem=problem,
            reasoning= "\n - " + "\n - ".join(reasoning),
            current_solution=current_solution
        )
        logger.info(f"RewardModel prompt:\n{reward_prompt}")
        
        messages = [
            {"role": "system", "content": "You are a reward model."},
            {"role": "user", "content": reward_prompt}
        ]
        
        reward_response = self.llm_client.generate_response(messages) 
        
        try:
            parsed_response = extract_and_parse_json(reward_response)
            score = parsed_response["score"]
            
            # Update stagnant_count based on score comparison
            if score <= self.previous_score:
                self.stagnant_count += 1
            else:
                self.stagnant_count = 0
            
            self.previous_score = score
            
            return json.dumps(parsed_response, indent=2)
        except json.JSONDecodeError:
            logger.error("Error: Reward model's response was not in valid JSON format. Returning raw response.")
            return reward_response

    def should_consult_sage(self):
        # This method remains unchanged
        return self.stagnant_count >= 1

class SwiftSage:
    def __init__(self, dataset, embeddings, prompt_template_dir, api_configs, use_retrieval=True, start_with_sage=True):
        prompt_template = PromptTemplate(prompt_template_dir)
        retrieval_augmentation = RetrievalAugmentation(dataset, embeddings) if use_retrieval else None
        
        sambanova_config = api_configs['SambaNova']
        swift_llm = LLMClient("Meta-Llama-3.1-8B-Instruct", sambanova_config)
        sage_llm = LLMClient("Meta-Llama-3.1-70B-Instruct", sambanova_config)
        reward_llm = LLMClient("Meta-Llama-3.1-8B-Instruct", sambanova_config)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.reward_model = RewardModel(prompt_template, reward_llm)
        self.start_with_sage = start_with_sage
    
    def solve(self, problem, max_iterations=10):
        logger.info(f"Starting to solve problem: {problem}")
        current_solution = "No current solution yet." # final answer
        current_reasoning = "No reasoning steps yet." # reasoning steps
        plan = "Initial plan: Take a deep breath and think step by step."
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback
        solved = False
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}")

            if (i == 0 and self.start_with_sage) or self.reward_model.should_consult_sage():
                sage_response = self.sage.generate_response(problem, current_reasoning, current_solution)
                sage_parsed = extract_and_parse_json(sage_response)
                critical_feedback = sage_parsed["critical_feedback"]
                plan = "\n".join(sage_parsed["revised_plan"])
                solved = sage_parsed["solved"].lower() == "true" if i != 0 else sage_parsed["solved"] 
                logger.info(f"Sage's feedback (iteration {i+1}):\n{critical_feedback}")
                logger.info(f"Sage's revised plan:\n{plan}")
            
            if not solved:
                swift_response = self.swift.generate_response(problem, current_reasoning, current_solution, plan, critical_feedback)
                swift_parsed = extract_and_parse_json(swift_response)
                current_solution = swift_parsed["final_answer"]
                reasoning = swift_parsed["reasoning_steps"]
                logger.info(f"Swift's reasoning:\n{reasoning}")
                logger.info(f"Swift's current solution:\n{current_solution}")
            
                reward_response = self.reward_model.calculate_reward(problem, reasoning, current_solution)
                reward_parsed = extract_and_parse_json(reward_response)
                score = reward_parsed["score"]
                feedback = reward_parsed["feedback"]
                logger.info(f"Reward for iteration {i+1}: {score}/10")
                logger.info(f"Feedback: {feedback}")
                critical_feedback = feedback
                current_reasoning = "\n -" + "\n - ".join(reasoning)
            
            if score >= 9 or solved:
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

s2 = SwiftSage(
    dataset, 
    embeddings, 
    prompt_template_dir, 
    api_configs, 
    use_retrieval=False,  # Set to False to disable retrieval augmentation
    start_with_sage=False  # Set to False to start with Swift
)

# problem = "Solve the equation: 2x + 5 = 13"
problem = "If h(x)=x-4 and g(h(x))=x^2-8x+10, find g(x)?"
reasoning, solution = s2.solve(problem)
logger.info(f"Final reasoning:\n")
for step in reasoning:
    logger.info(f" - {step}")
logger.info(f"Final solution:\n{solution}")

