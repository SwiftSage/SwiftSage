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
    def __init__(self, model_id, api_config, temperature=1.0, top_p=1.0, max_tokens=2048):
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

    ### Final Answer
    ```json
    {
    "reasoning_steps": [
        "Research and verify the current status (alive or deceased) of both Lowell Sherman and Jonathan Kaplan through reliable sources,
        Collect accurate birth and death dates (if applicable) for both individuals from multiple reliable sources,
        Cross-check the extracted data for consistency across different sources and evaluate the credibility of each source,
        Calculate the lifespan of each individual by subtracting their date of birth from their date of death (if applicable), or calculate the age of the living individual as of the current date,
        Address any potential challenges or conflicting information encountered during the research process by consulting additional sources or using alternative methods to verify the data,
        Compare the calculated lifespans (or ages) and conclude who lived longer, or is currently older if one or both individuals are still alive
    ],
    "final_answer": "Jonathan Kaplan lives longer than Lowell Sherman."
    }
    ```
    """     
    
    test_text = """
    ```json
{
  "feedback": "The current solution is mostly correct and well-structured. However, there is room for improvement in terms of ensuring the accuracy of the sources used. Additionally, the solution does not mention the reliability of the sources and the methods used to find the information. The clarity and efficiency of the steps is good, but a more concise presentation would make it even better. The efficiency of the solution is also good, as it directly compares the lifespans of the two individuals.",
  "score": 8
}
```
    """

    test_text = """

    ```json
{
  "reasoning_steps": [
    "Step 1: Define search parameters for Lowell Sherman and Jonathan Kaplan, including their full names, professions, and any relevant aliases or pseudonyms.",
    "Step 2: Identify a comprehensive list of reputable sources, including Wikipedia, IMDb, official biographies, and other reliable online sources. Prioritize primary sources, such as official documents and records, over secondary sources.",
    "Step 3: Search the identified sources for information on Lowell Sherman's and Jonathan Kaplan's birth and death dates, and document the results, including any inconsistencies or discrepancies.",
    "Step 4: Verify the accuracy of the information by cross-checking multiple sources and evaluating the credibility of each source. Consider the potential for errors, biases, or inconsistencies in the sources.",
    "Step 5: Handle potential ambiguities and inaccuracies by documenting conflicting information and using a systematic approach to resolve discrepancies, such as prioritizing primary sources or using additional verification methods.",
    "Step 6: Calculate the lifespan of Lowell Sherman and Jonathan Kaplan based on the verified birth and death dates, and compare the results to determine who lived longer.",
    "Step 7: Document the sources used to gather the information and provide a clear conclusion based on the verified data, including any limitations or potential biases in the analysis.",
    "Step 8: Consider additional factors that may have affected the lifespan of Lowell Sherman and Jonathan Kaplan, such as lifestyle, health conditions, or environmental factors, and discuss the potential implications of these factors on the results."
  ],
  "final_answer": "After conducting a thorough search and verifying the information, it appears that Lowell Sherman (1885-1934) lived for 49 years, while Jonathan Kaplan (1947-2023) lived for 76 years. Therefore, Jonathan Kaplan lives longer than Lowell Sherman."
]
```"""

    test_text = """
    {
  "reasoning_steps": [
    "Step 1: The inert pair effect is a phenomenon where the heavier elements in groups 13 and 15 tend to form ions with a charge of +3 rather than +5, due to the low-energy electrons in the s-orbital not participating in bonding. This is caused by the decrease in effective nuclear charge and the increase in atomic size, making it more difficult for the heavier elements to lose electrons and achieve a +5 oxidation state.",
    "Step 2: The trend of decreasing stability of the +5 oxidation state among the elements of Group 15 can be observed as follows: phosphorus pentoxide (P2O5), arsenic pentoxide (As2O5), antimony pentoxide (Sb2O5), and bismuth pentoxide (Bi2O5) exhibit a decrease in stability. This can be attributed to various thermodynamic properties such as enthalpy of formation, ease of reduction, and other experimental evidence and data.",
    "Step 3: Relativistic effects become more significant for heavier elements, leading to an increase in the atomic radius and a decrease in the effective nuclear charge. This results in a more pronounced inert pair effect and a decrease in stability of the +5 oxidation state. For example, the atomic radius increases from phosphorus (133 pm) to astatine (150 pm), and the electronegativity decreases from phosphorus (2.1) to astatine (0.6).",
    "Step 4: The trend of decreasing stability of the +5 oxidation state can be attributed to various factors such as the inert pair effect, increasing size of the atoms, relativistic effects, weakened bonds, and the increasing influence of d-subshells, electronegativity, and availability of d-orbitals for bonding.",
    "Step 5: The synthesis of the information from the previous steps provides a comprehensive explanation of the trend of decreasing stability of the +5 oxidation state among the elements of Group 15. This trend can be explained by the inert pair effect, increasing size of the atoms, relativistic effects, weakened bonds, and the increasing influence of d-subshells, electronegativity, and availability of d-orbitals for bonding. Some exceptions and limitations of this trend should be noted.",
    "Step 6: The final answer should be concise and directly address the question. The trend of decreasing stability of the +5 oxidation state among the elements of Group 15 can be attributed to the inert pair effect, increasing size of the atoms, relativistic effects, weakened bonds, and the increasing influence of d-subshells, electronegativity, and availability of d-orbitals for bonding, resulting in a decrease in stability from phosphorus to astatine."
  ],
  "final_answer": "The trend of decreasing stability of the +5 oxidation state among the elements of Group 15 is due to the inert pair effect, increasing size of the atoms, relativistic effects, weakened bonds, and the increasing influence of d-subshells, electronegativity, and availability of d-orbitals for bonding, resulting in a decrease in stability from phosphorus to astatine."""

    result = extract_and_parse_json(test_text)
    print(result)

    # print(extract_and_parse_json(test_text, ))
