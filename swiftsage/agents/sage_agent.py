import json
import logging

import numpy as np

from swiftsage.agents import Agent
from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class SageAgent(Agent):
    def __init__(self, prompt_template, llm_client):
        super().__init__(prompt_template, llm_client)

    def generate_response(self, prompt, reasoning, current_solution, prefill=None):
        if prefill is None:
            prefill = self.llm_client.support_prefill
            
        sage_prompt = self.prompt_template.format(
            "sage",
            prompt=prompt,
            reasoning=reasoning, 
            current_solution=current_solution
        )
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": sage_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<solved>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages)[0]
        if prefill and not self.llm_client.prefix_in_response:
            response = "<solved>" + response

        raw_messages = {
            "model_input": sage_prompt,
            "model_output": response
        }

        try:
            parsed_response = extract_and_parse_markup(response)
            return parsed_response, raw_messages
        except json.JSONDecodeError:
            logger.error("Error: Sage's response was not in valid JSON format. Returning raw response.")
            return response, raw_messages
