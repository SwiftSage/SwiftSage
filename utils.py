import datetime
import json
import logging
import os
import re
from abc import ABC, abstractmethod

import dirtyjson
import hjson
import numpy as np
import openai
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

api_configs = {
    "SambaNova": {
        "api_key": os.environ.get("SAMBANOVA_API_KEY"),
        "url_base": "https://api.sambanova.ai/v1"
    },
    "Together": {
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "url_base": "https://api.together.xyz/v1"
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

 

def extract_and_parse_markup(text):
    keys = ["reasoning_steps", "final_answer", "feedback", "score", "critical_feedback", "revised_plan", "solved", "plan", "code"]
    result = {}
    if "<final_answer>" in text and "</final_answer>" not in text:
        text = text + "</final_answer>"

    for key in keys:
        # Create a pattern for each key
        pattern = f'<{key}>(.*?)</{key}>'
        
        # Search for the pattern in the text
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # Extract the content, strip whitespace, and add to the result
            content = match.group(1).strip()
            result[key] = content

    if "code" in result.keys():
        result["code"] = result["code"].replace("```python", "").replace("```", "").strip()

    return result


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
        return template


class LLMClient:
    def __init__(self, model_id, api_config, temperature=0.3, top_p=1.0, max_tokens=3000, logger=None):
        self.client = openai.OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['url_base']
        )
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logger = logger

    def generate_response(self, messages):
        self.logger.info(f"Sending request to {self.model_id}")
        self.logger.info(f"Messages: {messages}")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        self.logger.info(f"Response from {self.model_id}:\n{content}")
        return content





if __name__ == "__main__":
    test_text = "test"
     
    print(extract_and_parse_markup(test_text))



"""

def extract_and_parse_json(text):

    keys_and_types = [
        ("reasoning_steps", list),
        ("final_answer", str),
        ("feedback", str),
        ("score", str),
        ("score", int),
        ("feedback", str),
        ("solved", str),
        ("critical_feedback", str),
        ("revised_plan", list),
    ]

    # Try to parse the JSON first
    try:
        # find the first and last curly braces and parse the json
        first_brace = text.find("{")
        last_brace = text.rfind("}")  
        if last_brace == -1:
            text = text + "\"}"
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            data = json.loads(text[first_brace:last_brace+1])
        return data
    except Exception as e:
        data = {}
        try: 
            data = dirtyjson.loads(text) 
        except Exception as e:
            pass
        # If JSON parsing fails, use regex to extract key-value pairs
        
        for key, _ in keys_and_types:
            # pattern = rf'"{key}"\s*:\s*([\[{{].*?[\]}}]|".*?")'
            pattern = rf'"{key}"\s*:\s*([\[{{].*?[\]}}]|".*?"|[-+]?\d+)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(1))
                except Exception as e:
                    value = match.group(1).strip('"')
                data[key] = value

    result = {}
    for key, expected_type in keys_and_types:
        if key in result.keys() and result[key] is not None:
            continue
        # Use fuzzy matching to find the closest key
        try:
            closest_key, score = process.extractOne(key, data.keys())
        except Exception as e:
            continue
        if score > 80:  # You can adjust this threshold
            value = data[closest_key]
            
            # Type checking and conversion
            if expected_type == list and isinstance(value, str):
                value = [item.strip() for item in value.strip('[]').split(',')]
            elif expected_type == str and isinstance(value, list):
                value = ', '.join(value)
            elif expected_type == int and value is not None:
                try:
                    value = int(value)
                except ValueError:
                    value = None
            
            result[key] = value
        else:
            result[key] = None 
    
    for key in list(result.keys()):
        if result[key] is None:
            del result[key]
    return result

def extract_and_parse_json_v1(text):
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

 
 

"""