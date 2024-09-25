import json
import logging

from swiftsage.agents import SwiftAgent, SageAgent, Feedback, RetrievalAugmentation
from swiftsage.utils import LLMClient, PromptTemplate, PythonExecutor


logger = logging.getLogger("SwiftSage")


class SwiftSage:
    def __init__(self, dataset, embeddings, prompt_template_dir, swift_config, sage_config, feedback_config, use_retrieval=True, start_with_sage=False):
        prompt_template = PromptTemplate(prompt_template_dir)
        retrieval_augmentation = RetrievalAugmentation(dataset, embeddings) if use_retrieval else None
        
        # add logger to the following LLMClient 
        swift_llm = LLMClient(**swift_config, logger=logger)
        sage_llm = LLMClient(**sage_config, logger=logger)
        feedback_llm = LLMClient(**feedback_config, logger=logger)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.feedback_model = Feedback(prompt_template, feedback_llm)
        self.start_with_sage = start_with_sage
        # self.executor = PythonExecutor(get_answer_from_stdout=True)
    
    def solve(self, problem, max_iterations=10, reward_threshold=8):
        messages = []
        
        def log_and_append(message):
            logger.info(message)
            messages.append(message)

        log_and_append(f"Starting to solve problem: {problem}")
        current_solution = "No current solution yet."  # final answer
        current_reasoning = "No reasoning steps yet."  # reasoning steps
        plan = "Initial plan: Take a deep breath and think step by step."
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback
        solved = False

        for i in range(max_iterations):
            log_and_append(f"Iteration {i+1}")

            #  Use the Sage Agent 
            if (i == 0 and self.start_with_sage) or self.feedback_model.should_consult_sage():
                sage_parsed = self.sage.generate_response(problem, current_reasoning, current_solution) 
                critical_feedback = sage_parsed["critical_feedback"]
                current_reasoning = sage_parsed["reasoning_steps"] 
                current_code = sage_parsed["code"] 

                solved = sage_parsed["solved"].lower() == "true" if i != 0 else sage_parsed["solved"] 
                if solved:
                    return current_reasoning, current_solution, messages

                log_and_append(f"Sage's feedback (iteration {i+1}):\n{critical_feedback}")
                log_and_append(f"Sage's reasoning steps:\n{current_reasoning}")
                self.sage.feedbacks[i] = critical_feedback
                
                # run the code 
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                log_and_append(f"Sage Code execution report: {code_report}")
                log_and_append(f"Sage Code execution result: {code_result}")
                current_reasoning = current_reasoning + f"\n\nThe generated code is:\n\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result
                
                log_and_append("Activated Sage, so we should return the reasoning and solution from Sage.")
                return current_reasoning, current_solution, messages
            
            if not solved:
                # Use the Swift Agent 
                swift_parsed = self.swift.generate_response(problem, current_reasoning, current_solution, plan, critical_feedback)
                
                if "code" not in swift_parsed and "final_answer" not in swift_parsed: 
                    log_and_append("Swift's response does not contain the 'final_answer' or 'code' field. Returning raw response.")
                    self.feedback_model.scores.append(0)
                    self.feedback_model.feedbacks.append("No feedback")
                    self.feedback_model.stagnant_count += max_iterations # force to use Sage Agent
                    continue 
                
                current_plan = swift_parsed["plan"]
                current_code = swift_parsed["code"]
                current_answer = swift_parsed.get("final_answer", None)

                self.swift.plans[i] = current_plan
                self.swift.codes[i] = current_code  

                log_and_append(f"Swift's plan:\n{current_plan}")
                log_and_append(f"Swift's code:\n{current_code}")

                # Call sandbox to run the code and get the result
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                if code_report != "Done":
                    # retry generate_response for swift 
                    log_and_append(f"Code execution report: {code_report}")
                    log_and_append("Code execution failed. Retrying the Swift agent.")
                    continue
                log_and_append(f"Code execution report: {code_report}")
                log_and_append(f"Code execution result: {code_result}")
            
                current_reasoning = current_plan + f"\nThe generated code is:\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result

                # Calling the reward model to provide feedback and score 
                reward_parsed = self.feedback_model.calculate_reward(problem, current_reasoning, current_solution)
                score = int(reward_parsed["score"])
                feedback = reward_parsed["feedback"] 
                prev_score = self.feedback_model.scores[-1] if len(self.feedback_model.scores) > 0 else 0
                self.feedback_model.scores.append(score)
                self.feedback_model.feedbacks.append(feedback)

                log_and_append(f"Reward for iteration {i+1}: {score}/10")
                log_and_append(f"Feedback: {feedback}")

                if False and score < prev_score:
                    log_and_append("Score is lower than the previous score. Stopping the iteration. Reverting to the previous solution and reasoning.")
                    # revert to the previous solution and reasoning
                    current_solution = self.swift.codes[i-1]
                    current_reasoning = self.swift.plans[i-1]
                    continue 

                critical_feedback = feedback 

            if score >= reward_threshold or solved:
                log_and_append("Perfect solution found!")
                return current_reasoning, current_solution, messages 
            
            if self.feedback_model.should_consult_sage():
                log_and_append("Reward model: The solution quality hasn't improved recently. Consulting Sage for the next iteration.")
        
        log_and_append("Max iterations reached without finding a perfect solution.")
        log_and_append("Problem solving completed")
        return current_reasoning, current_solution, messages
