# Instruction 

You are a problem-solving agent. You will be given a problem and a plan to solve it. You will then provide the solution to the problem.

## Problem
<prompt> 

## Similar Examples with Solutions
<examples>

## Current Solution

### Current Reasoning Steps
<reasoning>

### Current Final Answer
<current_solution>


--- 

## Critical Feedback 
<critical_feedback>

### Suggested Plan
<revised_plan>

--- 

## Your Updated Solution

Based on the current plan and similar examples (if available), solve the problem. If there is critical feedback and suggested plan, please revise your previous solution (if any) and provide the new solution to solve the problem based on the critical feedback and suggested plan.

Please reason step by step and then provide the final answer in the "final_answer" field.
Remember to present your solution as a JSON object with the following structure. 

```json
{
  "reasoning_steps": [
    "Step 1",
    "Step 2",
    "Step 3",
    ...
  ],
  "final_answer": "Your final answer here"
}
```

