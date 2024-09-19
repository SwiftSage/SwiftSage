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





api_configs = {
    "SambaNova": {
        "api_key": os.environ.get("SAMBANOVA_API_KEY"),
        "url_base": "https://api.sambanova.ai/v1"
    }
    # You can add more API configurations here for other providers
}

class Agent(ABC):
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    @abstractmethod
    def generate_response(self, prompt):
        pass


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
    def __init__(self, model_id, api_config, temperature=0.8, top_p=1.0):
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


logger = setup_logging()
