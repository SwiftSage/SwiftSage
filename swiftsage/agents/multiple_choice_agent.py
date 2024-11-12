import json
import logging
import re
import random

import numpy as np

from swiftsage.agents import Agent
from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class MultipleChoiceAgent(Agent):
    def __init__(self, prompt_template, llm_client):
        super().__init__(prompt_template, llm_client)

    def generate_response(self, question, plan, code, code_res, prefill=None):
        if prefill is None:
            prefill = self.llm_client.support_prefill
        
        mc_prompt = self.prompt_template.format(
            "multiple_choice",
            question=question,
            current_plan=plan,
            current_code=code,
            current_code_res=code_res,
        )

        messages = [
            {"role": "system", "content": ''},
            {"role": "user", "content": mc_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<choice>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages)[0]
        if prefill and not self.llm_client.prefix_in_response:
            response = "<choice>" + response

        raw_messages = {
            "model_input": mc_prompt,
            "model_output": response
        }
        
        try:
            parsed_responses = extract_and_parse_markup(response)
            if len(parsed_responses) != 1:
                # maybe A. (A) A) or "The answer is A"
                # extract the first capital letter from A to E from the response
                # and return the corresponding choice
                choice = re.search(r"[A-E]", response)
                if choice:
                    choice = choice.group().strip()
                    parsed_responses = {"choice": choice}
                else:
                    parsed_responses = {"choice": random.choice(["A", "B", "C", "D", "E"])}
            return parsed_responses, raw_messages
        except json.JSONDecodeError:
            logger.error("Error: Swift's response was not in valid JSON format. Returning raw response.")
            return response, raw_messages
