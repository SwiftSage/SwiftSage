import json
import logging

import numpy as np

from swiftsage.agents import Agent
from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class SageAgent(Agent):
    def __init__(self, prompt_template, llm_client):
        super().__init__(prompt_template, llm_client)
        self.feedbacks = {}
        self.plans = {}
        

    def generate_response(self, prompt, reasoning, current_solution, prefill=True):
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
        if prefill:
            messages.append({"role": "assistant", "content": "<solved>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages)
        # logger.info(f"SageAgent raw response:\n{response}")
        if prefill:
            response = "<solved>" + response
        try:
            parsed_response = extract_and_parse_markup(response)
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Sage's response was not in valid JSON format. Returning raw response.")
            return response
