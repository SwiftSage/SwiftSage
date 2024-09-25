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
        self.plans = {}
        self.codes = {}

    def generate_response(self, prompt, reasoning, current_solution, plan, critical_feedback, prefill=None):
        if prefill is None:
            prefill = self.llm_client.support_prefill
        logger.info("SwiftAgent generating response")
        if self.retrieval_augmentation:
            query_embedding = self.get_query_embedding(prompt)
            similar_examples = self.retrieval_augmentation.get_similar_examples(query_embedding)
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
            revised_plan=plan
        )
        # logger.info(f"SwiftAgent prompt:\n{swift_prompt}")

        messages = [
            {"role": "system", "content": ''},
            {"role": "user", "content": swift_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<plan>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages) 
        if prefill:
            response = "<plan>" + response
        
        try:
            parsed_response = extract_and_parse_markup(response)
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Swift's response was not in valid JSON format. Returning raw response.")
            return response

    def get_query_embedding(self, query):
        # Implement query embedding generation
        return np.random.rand(768)  # Placeholder, replace with actual embedding
 