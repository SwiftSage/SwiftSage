# Instruction

You are a reward model. You will be given a problem, an answer, and the reasoning process. You will then evaluate the answer based on the criteria provided.

## Problem
<problem>

### Current Reasoning Process
<reasoning>

## Current Answer
<current_solution>


## Your Evaluation

Your evaluation should include two parts.

### Correctness of the Current Answer

- You should see the current answer first, evaluate the correctness of the current answer and give the score.
    - score=0 means the provided answer is wrong
    - score=1 means the provided answer is correct
    - The correctness of the answer is independent with the reasoning process. Just see the answer and evaluate its correctness.

### Feedback for the Reasoning Process

- Carefully review the reasoning process and give your feedback.
- If the answer is wrong (score=0), please provide feedback by pointing out any mistakes or missing parts in the solution.
- Take care and do not give false information in the critical feedback.
- If you are not sure about the reasoning process, explain why you are not sure about the final answer.
- Note that you should not ask for any input from the console in the code. The code should be self-contained and print the final answer at the end.


## Output Format

Remember to present your output in the following format:

<score>
Whether the current answer is correct. 0 or 1.
</score>

<feedback>
Your critical feedback for the reasoning process here.
</feedback>

# Important Notes

You must follow the format strictly, do not miss any field. Start your output by "<score>" and end your output by "</feedback>".