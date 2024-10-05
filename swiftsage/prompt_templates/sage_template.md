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

We are not sure if the current solution is correct, can you provide a critical feedback on the current solution and suggest a revised plan for the next steps. Consider any challenges or improvements needed. 

If the solution and answer are correct, please set `solved` to `"True"`, and leave `critical_feedback` and `reasoning_steps` empty.
Please point out the errors in the current solution if there are any in the `critical_feedback` field, and then provide the revised plan in the `reasoning_steps` field, and finally provide the final answer in the `final_answer` field. Note that the code should not ask for any input from console, but it should be self-contained and print the final answer at the end.


Format your response in the following format: 


<solved>
[True or False]
</solved>

<critical_feedback>
[Your critical feedback here.]
</critical_feedback>

<reasoning_steps>
[Put your reasoning steps here to revise the previous solution. Use additional knowledge if needed and then we will write the code to solve the problem in the next field.]
</reasoning_steps>

<code>
[Put your updated code here to solve the problem.]
</code>
 

