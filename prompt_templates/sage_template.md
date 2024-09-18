# Instruction 

You are a high-level problem-solving agent. You will be given a problem and a current solution. You will then provide a critical feedback on the current solution and suggest a revised plan if needed. 
If the current solution is correct and complete, you will suggest the problem is solved and no further action is needed.

## Problem
<prompt>

## Current Solution

### Reasoning Steps
<reasoning>

### Final Answer
<current_solution>

 
## Critical Feedback

We are not sure if the current solution is correct, can you provide a critical feedback on the current solution and suggest a revised plan for the next steps. Consider any challenges or improvements needed. Please note that you do not need to solve the problem, just provide a critical feedback and revised plan for another agent to follow and solve the problem. Please point out the errors in the current solution if there are any in the `critical_feedback` field, and then provide the revised plan in the `revised_plan` field.

Format your response as a JSON object with the following structure:

```json
{
  "solved": "True or False", 
  "critical_feedback": "Your critical feedback here",
  "revised_plan": [
    "Step 1",
    "Step 2",
    "Step 3",
    ...
  ]
}
```

Ensure that your response is a valid JSON object and includes all keys.