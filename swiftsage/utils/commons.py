import datetime
import json
import logging
import os
import re

import openai
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


api_configs = {
    "OpenAI": {
        "provider": "OpenAI",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "url_base": os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
        "support_prefill": False
    },
    "SambaNova": {
        "provider": "SambaNova",
        "api_key": os.environ.get("SAMBANOVA_API_KEY"),
        "url_base": "https://api.sambanova.ai/v1",
        "support_prefill": False
    },
    "Together": {
        "provider": "Together",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "url_base": "https://api.together.xyz/v1",
        "support_prefill": True,
        "prefix_in_response": False,
    },
    "Groq":{
        "provider": "Groq",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "url_base": "GROQ",
        "support_prefill": True,
        "prefix_in_response": False,
    },
    "vLLM": {
        "provider": "vLLM",
        "api_key": "token-abc123",
        "url_base": "http://localhost:8000/v1",
        "support_prefill": True,
        "prefix_in_response": True,
    },
    # You can add more API configurations here for other providers
}


def setup_logging():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/swiftsage_log_{timestamp}.txt"
    # create folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
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
    if "<score>" in text and "</score>" not in text:
        text = text + "</score>"

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
        # find the first full code block inside ```python and ``` and extract the code if any 
        if "```python" in result["code"]:
            code_block_pattern = r'```python\s*([\s\S]*?)\s*```'
            code_blocks = re.findall(code_block_pattern, result["code"], re.IGNORECASE)
            if code_blocks:
                result["code"] = code_blocks[0]
        
        # result["code"] = result["code"].replace("```python", "").replace("```", "").strip()

    return result


class PromptTemplate:
    def __init__(self, template_dir):
        self.template_dir = template_dir
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        for filename in ['swift_template.md', 'sage_template.md', 'feedback_template.md', 'multiple_choice_template.md']:
            with open(os.path.join(self.template_dir, filename), 'r') as f:
                key = filename[:-len("_template.md")]
                self.templates[key] = f.read()

    def format(self, key, **kwargs):
        template = self.templates.get(key, "")
        for k, v in kwargs.items():
            template = template.replace("<" + k + ">", str(v)) 
        return template


class LLMClient:
    def __init__(self, model_id, api_config, temperature=0.3, top_p=1.0, max_tokens=2048, logger=None):
        if api_config['url_base'] == "GROQ":
            self.client = Groq(api_key=api_config['api_key'])
        else:
            # openai api
            self.client = openai.OpenAI(
                api_key=api_config['api_key'],
                base_url=api_config['url_base']
            )
        self.api_provider = api_config['provider']
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logger = logger
        self.support_prefill = api_config.get("support_prefill", False)
        self.prefix_in_response = api_config.get("prefix_in_response", False)

    def generate_response(self, messages, n=1):
        self.logger.debug(f"Sending request to {self.model_id}")
        self.logger.debug(f"Messages: {messages}")

        if self.api_provider == "vLLM" and self.support_prefill:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=n,
                extra_body={
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=n,
            )
            
        content = [response.choices[i].message.content for i in range(n)]
        self.logger.debug(f"Response from {self.model_id}:\n{content}")
        return content


if __name__ == "__main__":
    test_text = """
<plan> \nTo solve this problem, we need to find all possible values of $n$ given that the roots of the polynomial $x^2 - mx + n$ are positive prime integers (not necessarily distinct) and $m < 20.$\n\nWe can use Vieta's formulas, which state that for a quadratic polynomial $ax^2 + bx + c = 0$ with roots $r_1$ and $r_2,$ we have $r_1 + r_2 = -\\frac{b}{a}$ and $r_1r_2 = \\frac{c}{a}.$\n\nIn our case, we have $a = 1,$ $b = -m,$ and $c = n.$ Therefore, we have $r_1 + r_2 = m$ and $r_1r_2 = n.$\n\nSince $r_1$ and $r_2$ are positive prime integers, we know that $r_1 + r_2$ must be even, since the sum of two odd numbers is even, and the sum of an odd number and an even number is odd.\n\nSince $m < 20,$ we can list out all possible values of $m$ that are even and less than 20, which are $2, 4, 6, 8, 10, 12, 14, 16, 18.$\n\nFor each value of $m,$ we can find the corresponding values of $r_1$ and $r_2$ by finding two positive prime integers that add up to $m.$ We can then calculate the corresponding value of $n$ by multiplying $r_1$ and $r_2$ together.\n\nWe can then count the number of distinct values of $n$ that we find.\n\n</plan>\n\n<code>\nimport math\n\ndef is_prime(n):\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    sqrt_n = math.isqrt(n)\n    for i in range(3, sqrt_n + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\ndef find_n_values():\n    n_values = set()\n    for m in range(2, 20, 2):\n        for r1 in range(2, m):\n            r2 = m - r1\n            if is_prime(r1) and is_prime(r2):\n                n = r1 * r2\n                n_values.add(n)\n    return len(n_values)\n\nresult = find_n_values()\nprint(result)\n</code>
    """
     
    res = extract_and_parse_markup(test_text)
    print('test')
 