import datetime
import json
import logging
import os
import re

import dirtyjson
import hjson
import numpy as np
import openai
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

api_configs = {
    "SambaNova": {
        "api_key": os.environ.get("SAMBANOVA_API_KEY"),
        "url_base": "https://api.sambanova.ai/v1",
        "support_prefill": False
    },
    "Together": {
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "url_base": "https://api.together.xyz/v1",
        "support_prefill": True
    }
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
        for filename in ['swift_template.md', 'sage_template.md', 'feedback_template.md']:
            with open(os.path.join(self.template_dir, filename), 'r') as f:
                key = filename.split('_')[0]
                self.templates[key] = f.read()

    def format(self, key, **kwargs):
        template = self.templates.get(key, "")
        for k, v in kwargs.items():
            template = template.replace("<" + k + ">", str(v)) 
        return template


class LLMClient:
    def __init__(self, model_id, api_config, temperature=0.3, top_p=1.0, max_tokens=2048, logger=None):
        self.client = openai.OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['url_base']
        )
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logger = logger
        self.support_prefill = api_config.get("support_prefill", False)

    def generate_response(self, messages):
        self.logger.info(f"Sending request to {self.model_id}")
        self.logger.info(f"Messages: {messages}")
        print(f"Sending request to {self.model_id}")
        print(f"Messages: {messages}")

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        # debug print
        print(f"model_id: {self.model_id}")
        print(f"max_tokens: {self.max_tokens}")
        content = response.choices[0].message.content
        self.logger.info(f"Response from {self.model_id}:\n{content}")
        return content





if __name__ == "__main__":
    test_text = """
<code>
```python
num1 = 9.11
num2 = 9.8

if num1 > num2:
    print("9.11 is larger.")
elif num2 > num1:
    print("9.8 is larger.")
else:
    print("Both numbers are equal.")
```
Alternatively, a more concise version:
```python
print("9.11 is larger" if 9.11 > 9.8 else "9.8 is larger")
```
</code>
    """
     
    print(extract_and_parse_markup(test_text))

 