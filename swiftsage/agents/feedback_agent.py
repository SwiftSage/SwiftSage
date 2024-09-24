import json
import logging

from swiftsage.utils.commons import extract_and_parse_markup


logger = logging.getLogger("SwiftSage")


class Feedback:
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client 
        self.scores = [] 
        self.feedbacks = [] 
        self.stagnant_count = 0

    def calculate_reward(self, problem, reasoning, current_solution, prefill=True):
        reward_prompt = self.prompt_template.format(
            "reward",
            problem=problem,
            reasoning= reasoning,
            current_solution=current_solution
        )
        # logger.info(f"Feedback prompt:\n{reward_prompt}")
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": reward_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": "<feedback>"}) # prefix-filling 
        
        reward_response = self.llm_client.generate_response(messages) 
        if prefill:
            reward_response = "<feedback>" + reward_response
        
        try:
            parsed_response = extract_and_parse_markup(reward_response)
            score = int(parsed_response["score"])
            
            # Update stagnant_count based on score comparison
            if len(self.scores) > 0 and score <= self.scores[-1]:
                self.stagnant_count += 1
            else:
                self.stagnant_count = 0
            
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Error: Reward model's response was not in valid JSON format. Returning raw response.")
            return reward_response

    def should_consult_sage(self):
        # This method remains unchanged
        return self.stagnant_count >= 1 or (len(self.scores) > 0 and self.scores[-1] < 5)
