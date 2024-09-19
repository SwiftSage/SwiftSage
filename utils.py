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

from fuzzywuzzy import process
import dirtyjson



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
    keys = ["reasoning_steps", "final_answer", "feedback", "score", "critical_feedback", "revised_plan", "solved"]
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

    return result


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
    def __init__(self, model_id, api_config, temperature=0.3, top_p=1.0, max_tokens=3000):
        self.client = openai.OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['url_base']
        )
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate_response(self, messages):
        logger.info(f"Sending request to {self.model_id}")
        logger.info(f"Messages: {messages}")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        logger.info(f"Response from {self.model_id}:\n{content}")
        return content


logger = setup_logging()


if __name__ == "__main__":
    test_text = """
    
 <reasoning_steps>

Let's denote the number of band members in each row as $m$ and the number of rows as $r$. Given that the total number of band members is less than 100 and that there are two members left over when arranged in the rectangular formation with $m$ members in each row and $r$ rows, we can express this as $mr + 2$. 

Now, when the director increases the number of members in each row by 1 and reduces the number of rows by 2, we know that there are exactly enough places in the new formation for each band member. This can be represented by the equation $(m + 1)(r - 2)$.

We can now set these two expressions equal to each other since they both represent the total number of band members:

$mr + 2 = (m + 1)(r - 2)$.

Expanding the right side gives us:

$mr + 2 = mr - 2m + r - 2$

After canceling out $mr$ from both sides, we have:

$2 = -2m + r - 2$

Simplifying further, we get:

$r = 2m + 4$

Now we need to find the largest number of members the band could have. We also need to remember that this number must be less than 100.

Given the expression for $r$ as $2m + 4$, we know that the total number of band members is $mr + 2$. Substitute the expression for $r$ to get $m(2m + 4) + 2$. Simplifying this gives us:

$2m^2 + 4m + 2$.

This is a quadratic expression that we want to maximize. Since the maximum occurs at the vertex, we want to find the vertex of this quadratic. 

The x-coordinate of the vertex of a parabola $ax^2 + bx + c$ is given by $\frac{-b}{2a}$. In this case, $a = 2$ and $b = 4$.

Plugging these values into the formula for the x-coordinate of the vertex gives us:

$\frac{-4}{2*2} = -1$

Since $m$ is an integer, $m$ must be less than $-1$, which is not possible because there are no negative band members in each row. However, this problem likely assumes that $m$ will be a positive number, since we're counting band members.

If $m$ is a positive number, we can disregard the value of $m$ for which we find the vertex. Instead, let's plug in integers into $m$ until we get a product greater than or equal to 98, which is one less than 100, and still less than 100.

The largest $mr + 2$ less than 100 occurs when $m$ equals 6 and $r$ equals 16:

$6*16 + 2 = 98$

However, we could verify that when $m$ equals 7 and $r$ equals 18, the product also produces a number less than 100:

$7*18 + 2 = 128$

Let’s verify if this makes sense.

</reasoning_steps>

<final_answer>
98
</final_answer> """
    print(extract_and_parse_markup(test_text))