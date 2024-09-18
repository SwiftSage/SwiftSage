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

def setup_logging():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/swiftsage_log_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='w'
    )
    
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger('SwiftSage')

logger = setup_logging()

 

def extract_and_parse_json(text):
    def find_json_objects(s):
        # Find all substrings that look like JSON objects
        json_like_strs = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', s)
        return json_like_strs

    def try_parse_json(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                s = s.replace("\n", "")
                return hjson.loads(s)
            except json.JSONDecodeError:
                return None
            return None 

    # First, try to find JSON within code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_blocks = re.findall(code_block_pattern, text, re.IGNORECASE)
    
    all_json_candidates = []
    
    # Add JSON candidates from code blocks
    for block in code_blocks:
        all_json_candidates.extend(find_json_objects(block))
    
    # Add JSON candidates from the entire text
    all_json_candidates.extend(find_json_objects(text))
    
    # Sort candidates by length, descending
    all_json_candidates.sort(key=len, reverse=True)
    
    # Try to parse each candidate
    for candidate in all_json_candidates:
        parsed_json = try_parse_json(candidate)
        if parsed_json is not None:
            return parsed_json
    
    raise ValueError("No valid JSON object found in the text")

 
 
class PromptTemplate:
    def __init__(self, template_dir):
        self.template_dir = template_dir
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        for filename in ['swift_template.md', 'sage_template.md', 'reward_template.md']:
            with open(os.path.join(self.template_dir, filename), 'r') as f:
                key = filename.split('_')[0]
                self.templates[key] = f.read()

    def format(self, key, **kwargs):
        template = self.templates.get(key, "")
        for k, v in kwargs.items():
            template = template.replace("<" + k + ">", str(v))
        # logger.info(f"Formatted {key} template:\n{template}")
        return template


class LLMClient:
    def __init__(self, model_id, api_config, temperature=0.7, top_p=1.0):
        self.client = openai.OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['url_base']
        )
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, messages):
        logger.info(f"Sending request to {self.model_id}")
        logger.info(f"Messages: {messages}")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        content = response.choices[0].message.content
        logger.info(f"Response from {self.model_id}:\n{content}")
        return content


class Agent(ABC):
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    @abstractmethod
    def generate_response(self, prompt):
        pass

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
                sage_response = self.sage.generate_response(problem, current_reasoning, current_solution, plan)
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

# Usage
dataset = ["Example problem 1: ...", "Example problem 2: ...", "Example problem 3: ..."]
embeddings = np.random.rand(len(dataset), 768)  # Placeholder, replace with actual embeddings
prompt_template_dir = "./prompt_templates"

api_configs = {
    "SambaNova": {
        "api_key": os.environ.get("SAMBANOVA_API_KEY"),
        "url_base": "https://api.sambanova.ai/v1"
    }
    # You can add more API configurations here for other providers
}

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

