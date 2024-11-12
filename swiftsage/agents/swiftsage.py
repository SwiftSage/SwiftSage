import logging
import re
from collections import Counter

from swiftsage.agents import SwiftAgent, SageAgent, Feedback, RetrievalAugmentation, MultipleChoiceAgent
from swiftsage.utils import LLMClient, PromptTemplate, PythonExecutor


logger = logging.getLogger("SwiftSage")


class SwiftSage:
    def __init__(
        self, 
        prompt_template_dir, 
        swift_config, 
        sage_config, 
        feedback_config,
        multiple_choice=False,
        multiple_choice_config=None,
        only_swift=False,
        best_of_n=1,
        use_retrieval=False, 
        retrieval_dataset=None, 
        embeddings=None, 
        embedding_model=None,
    ):
        prompt_template = PromptTemplate(prompt_template_dir)
        
        retrieval_augmentation = RetrievalAugmentation(embedding_model, retrieval_dataset, embeddings) if use_retrieval else None
        
        # add logger to the following LLMClient 
        swift_llm = LLMClient(**swift_config, logger=logger)
        sage_llm = LLMClient(**sage_config, logger=logger)
        feedback_llm = LLMClient(**feedback_config, logger=logger)

        self.swift = SwiftAgent(prompt_template, swift_llm, retrieval_augmentation)
        self.sage = SageAgent(prompt_template, sage_llm)
        self.feedback_model = Feedback(prompt_template, feedback_llm)
        
        self.multiple_choice = multiple_choice
        if multiple_choice:
            multiple_choice_llm = LLMClient(**multiple_choice_config, logger=logger)
            self.multiple_choice_agent = MultipleChoiceAgent(prompt_template, multiple_choice_llm)

        self.only_swift = only_swift
        self.swift_n = best_of_n

        self.messages = {}
        self.raw_messages = {}

    def reset(self):
        self.feedback_model.reset()
        self.messages = {}
        self.raw_messages = {}

    def parse_and_execute(self, parsed_response):
        current_plan = parsed_response.get("plan", None)
        current_code = parsed_response.get("code", None)

        if not current_code:
            return {
            "plan": current_plan or "No plan provided in the response.",
            "code": "No code provided in the response.",
            "code_report": "No code provided in the response.",
            "solution": None,
        }

        # Call sandbox to run the code and get the result
        # add a timeout to the executor
        executor = PythonExecutor(get_answer_from_stdout=True, timeout=5)
        code_result, code_report = executor.apply(current_code)

        if code_report != "Done":
            return {
                "plan": current_plan,
                "code": current_code,
                "code_report": f"Code execution failed. The error message is: \n{code_report}",
                "solution": None,
            }

        match = re.search(r'\bis[:\s]*(.*)$', code_result)
        if match:
            code_result = match.group(1).strip()

        return {
            "plan": current_plan,
            "code": current_code,
            "code_report": code_report,
            "solution": code_result,
        }
        
    def solve(self, problem, max_iterations=10, reward_threshold=1):
        self.reset()

        current_reasoning = "No reasoning steps yet."  # reasoning steps
        current_solution = "No current solution yet."  # final answer
        critical_feedback = "No critical feedback yet."  # Initialize critical_feedback

        # Use the Swift Agent and refine the solution
        for i in range(max_iterations):
            if self.feedback_model.should_consult_sage():
                break

            # Best-of-N for Swift
            swift_parsed_list, raw_swift_messages = self.swift.generate_response(problem, current_reasoning, current_solution, critical_feedback, n=self.swift_n)
            self.raw_messages[f"Swift {i+1}"] = raw_swift_messages

            swift_res_list = []
            for swift_parsed in swift_parsed_list:
                swift_res_list.append(self.parse_and_execute(swift_parsed))

            cur_solutions = [res["solution"] for res in swift_res_list]
            filtered_solutions = [res for res in cur_solutions if res is not None]

            # N swift all failed
            if len(filtered_solutions) == 0:
                swift_res = swift_res_list[0]
                current_reasoning = swift_res['plan'] + f"\nThe generated code is:\n```python\n{swift_res['code']}\n```"
                current_solution = "No current solution yet."
                critical_feedback = swift_res["code_report"]
                self.feedback_model.scores.append(0)

                self.messages[f"Swift {i+1}"] = swift_res
                self.messages[f"Feedback {i+1}"] = {
                    "score": 0,
                    "feedback": critical_feedback
                }

                if self.only_swift:
                    return current_reasoning, current_solution, self.messages, self.raw_messages

                continue

            if self.multiple_choice:
                mc_solutions = []
                raw_mc_messages = []
                filtered_swift_res_list = [res for res in swift_res_list if res["solution"] in filtered_solutions]
                # Call the multiple choice agent to choose the best solution
                for res in filtered_swift_res_list:
                    choice, raw_mc_message = self.multiple_choice_agent.generate_response(problem, res["plan"], res["code"], res["solution"])
                    mc_solutions.append(choice['choice'])
                    raw_mc_messages.append(raw_mc_message)
                raw_mc_messages = {
                    "model_input": raw_mc_messages[0]["model_input"],
                    "model_output": [msg["model_output"] for msg in raw_mc_messages]
                }
                self.raw_messages[f"MultipleChoice {i+1}"] = raw_mc_messages

                assert len(mc_solutions) == len(filtered_swift_res_list)
                solution_counts = Counter(mc_solutions)
                swift_solution, _ = solution_counts.most_common(1)[0]
                swift_res = filtered_swift_res_list[mc_solutions.index(swift_solution)]

                current_reasoning = swift_res['plan'] + f"\nThe generated code is:\n```python\n{swift_res['code']}\n```"
                current_solution = swift_solution

                self.messages[f"Swift {i+1}"] = swift_res
                self.messages[f"Code {i+1}"] = cur_solutions
                self.messages[f"MultipleChoice {i+1}"] = mc_solutions
            else:
                # majority voting
                solution_counts = Counter(filtered_solutions)
                swift_solution, _ = solution_counts.most_common(1)[0]
                swift_res = swift_res_list[cur_solutions.index(swift_solution)]

                current_reasoning = swift_res['plan'] + f"\nThe generated code is:\n```python\n{swift_res['code']}\n```"
                current_solution = swift_res['solution']

                self.messages[f"Swift {i+1}"] = swift_res
                self.messages[f"Code {i+1}"] = cur_solutions

            if self.only_swift:
                return current_reasoning, current_solution, self.messages, self.raw_messages

            # Calling the reward model to provide feedback and score
            reward_parsed, raw_feedback_message = self.feedback_model.calculate_reward(problem, current_reasoning, current_solution)
            score = int(reward_parsed.get("score", 0))
            critical_feedback = reward_parsed.get("feedback", None)
            self.feedback_model.scores.append(score)

            self.messages[f"Feedback {i+1}"] = {
                "score": score,
                "feedback": critical_feedback
            }
            self.raw_messages[f"Feedback {i+1}"] = raw_feedback_message

            if score >= reward_threshold:
                return current_reasoning, current_solution, self.messages, self.raw_messages
            
        # if the loop ends without reaching the reward_threshold, consult the Sage Agent
        sage_parsed, raw_sage_message = self.sage.generate_response(problem, current_reasoning, current_solution) 

        self.raw_messages[f"Sage"] = raw_sage_message

        solved = sage_parsed.get("solved", "False").lower() == "true"
        critical_feedback = sage_parsed.get("critical_feedback", None)
        current_reasoning = sage_parsed.get("reasoning_steps", None)
        current_code = sage_parsed.get("code", None)

        self.messages["Sage"] = {
            "solved": solved,
            "feedback": critical_feedback,
            "plan": current_reasoning,
            "code": current_code,
            "code_report": None,
            "solution": current_solution,
        }

        # if the swift's answer is correct, directly return the answer
        if solved or current_code is None:
            return current_reasoning, current_solution, self.messages, self.raw_messages
        
        # run the code 
        executor = PythonExecutor(get_answer_from_stdout=True)
        code_result, code_report = executor.apply(current_code)

        self.messages[f"Sage"]["code_report"] = code_report
        self.messages[f"Sage"]["solution"] = code_result

        current_reasoning = current_reasoning + f"\n\nThe generated code is:\n\n```python\n{current_code}\n```"
        current_solution = code_result
        
        return current_reasoning, current_solution, self.messages, self.raw_messages
