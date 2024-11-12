import json
import logging

import numpy as np

from swiftsage.agents import Agent
from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class SwiftAgent(Agent):
    def __init__(self, prompt_template, llm_client, retrieval_augmentation=None):
        super().__init__(prompt_template, llm_client)
        self.retrieval_augmentation = retrieval_augmentation

    def generate_response(self, prompt, reasoning, current_solution, critical_feedback, n=1, prefill=None):
        if prefill is None:
            prefill = self.llm_client.support_prefill
        logger.debug("SwiftAgent generating response")
        if self.retrieval_augmentation:
            similar_examples = self.retrieval_augmentation.get_similar_examples(prompt)
            examples_text = "\n".join(similar_examples) # TODO: add more context to the prompt
        else:
            examples_text = "No similar examples available."
        
        swift_prompt = self.prompt_template.format(
            "swift",
            prompt=prompt,
            current_reasoning=reasoning, # TODO: check if this is needed
            examples=examples_text,
            current_solution=current_solution,
            critical_feedback=critical_feedback,
        )

        messages = [
            {"role": "system", "content": ''},
            {"role": "user", "content": swift_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<plan>"}) # prefix-filling 
        
        responses = self.llm_client.generate_response(messages, n=n)
        if prefill and not self.llm_client.prefix_in_response:
            responses = ["<plan>" + response for response in responses]

        raw_messages = {
            "model_input": swift_prompt,
            "model_output": [response for response in responses]
        }
        
        try:
            parsed_responses = [extract_and_parse_markup(response) for response in responses]
            return parsed_responses, raw_messages
        except json.JSONDecodeError:
            logger.error("Error: Swift's response was not in valid JSON format. Returning raw response.")
            return responses, raw_messages
