# Instruction

You are a reward model. You will be given a problem, a solution. You will then evaluate the solution based on the criteria provided.

## Problem
<problem>

## Current Solution

### Reasoning Steps
<reasoning>

### Final Answer
<current_solution>


## Your Evaluation

We are not sure if the current solution is correct. Please evaluate the current solution based on the following criteria:

1. Correctness
2. Completeness
3. Clarity
4. Efficiency

Provide a score from 1 to 10 and a brief explanation. 
If you are not sure about the final answer, provide a score between 1 to 7 and explain why you are not sure about the final answer.
Take care and do not give false information in the critical feedback.


## Output Format

Format your response as a JSON object with the following structure:

```json
{
  "feedback": "Your critical feedback here for analyzing the solution",
  "score": "[1-10]"
}
```

Ensure that your response is a valid JSON object and includes all keys.