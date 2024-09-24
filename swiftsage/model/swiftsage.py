import json
import logging

from swiftsage.model import SwiftAgent, SageAgent, FeedbackModel, RetrievalAugmentation
from swiftsage.utils import LLMClient, PromptTemplate, PythonExecutor


logger = logging.getLogger("SwiftSage")


class SwiftSage:
    def __init__(self, dataset, embeddings, prompt_template_dir, swift_config, sage_config, reward_config, use_retrieval=True, start_with_sage=False):
        prompt_template = PromptTemplate(prompt_template_dir)
        retrieval_augmentation = RetrievalAugmentation(dataset, embeddings) if use_retrieval else None
        
        # add logger to the following LLMClient 
        swift_llm = LLMClient(**swift_config, logger=logger)
        sage_llm = LLMClient(**sage_config, logger=logger)
        reward_llm = LLMClient(**reward_config, logger=logger)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.feedback_model = FeedbackModel(prompt_template, reward_llm)
        self.start_with_sage = start_with_sage
        # self.executor = PythonExecutor(get_answer_from_stdout=True)
    
    def solve(self, problem, max_iterations=10, reward_threshold=8):
        logger.info(f"Starting to solve problem: {problem}")
        current_solution = "No current solution yet." # final answer
        current_reasoning = "No reasoning steps yet." # reasoning steps
        plan = "Initial plan: Take a deep breath and think step by step."
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback
        solved = False
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}")
            

            #  Use the Sage Agent 
            if (i == 0 and self.start_with_sage) or self.feedback_model.should_consult_sage():
                sage_parsed = self.sage.generate_response(problem, current_reasoning, current_solution) 
                critical_feedback = sage_parsed["critical_feedback"]
                # plan = "\n - " + "\n - ".join(sage_parsed["revised_plan"])
                current_reasoning = sage_parsed["reasoning_steps"] 
                current_code = sage_parsed["code"] 

                solved = sage_parsed["solved"].lower() == "true" if i != 0 else sage_parsed["solved"] 
                if solved:
                    return current_reasoning, current_solution
                logger.info(f"Sage's feedback (iteration {i+1}):\n{critical_feedback}")
                # logger.info(f"Sage's reasoning steps:\n{current_reasoning}")
                self.sage.feedbacks[i] = critical_feedback
                
                # run the code 
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                logger.info(f"Sage Code execution report: {code_report}")
                logger.info(f"Sage Code execution result: {code_result}")
                current_reasoning = current_reasoning + f"\n\nThe generated code is:\n\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result
                
                # current_solution = sage_parsed["final_answer"]
                logger.info("Activated Sage, so we should return the reasoning and solution from Sage.")
                return current_reasoning, current_solution
            
            if not solved:
                # Use the Swift Agent 
                swift_parsed = self.swift.generate_response(problem, current_reasoning, current_solution, plan, critical_feedback)
                
                if "code" not in swift_parsed and "final_answer" not in swift_parsed: 
                    logger.info("Swift's response does not contain the 'final_answer' or 'code' field. Returning raw response.")
                    self.feedback_model.scores.append(0)
                    self.feedback_model.feedbacks.append("No feedback")
                    self.feedback_model.stagnant_count += max_iterations # force to use Sage Agent
                    continue 
                
                current_plan = swift_parsed["plan"]
                current_code = swift_parsed["code"]
                current_answer = swift_parsed.get("final_answer", None)

                self.swift.plans[i] = current_plan
                self.swift.codes[i] = current_code  

                logger.info(f"Swift's plan:\n{current_plan}")
                logger.info(f"Swift's code:\n{current_code}")

                # Call sandbox to run the code and get the result
                executor = PythonExecutor(get_answer_from_stdout=True)
                code_result, code_report = executor.apply(current_code)
                logger.info(f"Code execution report: {code_report}")
                logger.info(f"Code execution result: {code_result}")
            
                current_reasoning = current_plan + f"\nThe generated code is:\n```python\n{current_code}\n```"
                current_solution = "Answer (from running the code):\n " + code_result

                # Calling the reward model to provide feedback and score 
                reward_parsed = self.feedback_model.calculate_reward(problem, current_reasoning, current_solution)
                score = int(reward_parsed["score"])
                feedback = reward_parsed["feedback"] 
                prev_score = self.feedback_model.scores[-1] if len(self.feedback_model.scores) > 0 else 0
                self.feedback_model.scores.append(score)
                self.feedback_model.feedbacks.append(feedback)

                # detect if the score is lower than the previous score
                logger.info(f"Reward for iteration {i+1}: {score}/10")
                logger.info(f"Feedback: {feedback}")

                if False and score < prev_score:
                    logger.info("Score is lower than the previous score. Stopping the iteration. Reverting to the previous solution and reasoning.")
                    # revert to the previous solution and reasoning
                    current_solution = self.swift.codes[i-1]
                    current_reasoning = self.swift.plans[i-1]
                    continue 

                
                critical_feedback = feedback 

            
            if score >= reward_threshold or solved:
                logger.info("Perfect solution found!")
                return current_reasoning, current_solution 
            
            
            if self.feedback_model.should_consult_sage():
                logger.info("Reward model: The solution quality hasn't improved recently. Consulting Sage for the next iteration.")
        
        logger.info("Max iterations reached without finding a perfect solution.")
        logger.info("Problem solving completed")
        return current_reasoning, current_solution
