import json
import logging

from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class Feedback:
    def __init__(self, prompt_template, llm_client, K=2):
        self.prompt_template = prompt_template
        self.llm_client = llm_client 
        self.scores = []
        self.stagnant_count = 0
        self.K = K  # Number of stagnant scores before consulting Sage

    def calculate_reward(self, problem, reasoning, current_solution, prefill=None):
        if prefill is None:
            prefill = self.llm_client.support_prefill
        feedback_prompt = self.prompt_template.format(
            "feedback",
            problem=problem,
            reasoning=reasoning,
            current_solution=current_solution
        )
        # logger.info(f"Feedback prompt:\n{feedback_prompt}")
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": feedback_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<score>"}) # prefix-filling 
        
        response = self.llm_client.generate_response(messages)[0]
        if prefill and not self.llm_client.prefix_in_response:
            response = "<score>" + response
        
        raw_messages = {
            "model_input": feedback_prompt,
            "model_output": response
        }

        try:
            parsed_response = extract_and_parse_markup(response)
            score = int(parsed_response.get("score", -1)) # -1 means no score was found
            
            # Update stagnant_count based on score comparison
            if len(self.scores) > 0 and score <= self.scores[-1]:
                self.stagnant_count += 1
            else:
                self.stagnant_count = 0
            
            return parsed_response, raw_messages
        except json.JSONDecodeError:
            logger.error("Error: Reward model's response was not in valid JSON format. Returning raw response.")
            return response, raw_messages

    def should_consult_sage(self):
        # This method remains unchanged
        return self.stagnant_count >= self.K # or (len(self.scores) > 0 and self.scores[-1] < 5)
    
    def reset(self):
        self.scores = []
        self.stagnant_count = 0
