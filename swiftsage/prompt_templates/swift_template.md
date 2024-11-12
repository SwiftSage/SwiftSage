# Instruction 

## Similar Examples with Solutions

### Example Task 0

<task>
Hi, how are you?
</task>

<plan>
Step 1: Greet the user.
Step 2: Ask how you can help.
</plan>
 
<code> 
print("Hi, I'm doing well. How are you?")
print("How can I help you?")
</code>

### Example Task 1
 
<task>
Convert the point $(0, -3 \sqrt{3}, 3)$ in rectangular coordinates to spherical coordinates.  Enter your answer in the form $(\rho,\theta,\phi),$ where $\rho > 0,$ $0 \le \theta < 2 \pi,$ and $0 \le \phi \le \pi.$
</task>

<plan>
Step 1. Recall the formulas for converting from rectangular coordinates $(x, y, z)$ to spherical coordinates $(\rho, \theta, \phi)$:
   - $\rho = \sqrt{x^2 + y^2 + z^2}$
   - $\theta = \arctan2(y, x)$
   - $\phi = \arccos\left(\frac{z}{\rho}\right)$

Step 2. Given point: $(0, -3\sqrt{3}, 3)$
   $x = 0$
   $y = -3\sqrt{3}$
   $z = 3$

Step 3. Calculate $\rho$ using the formula.

Step 4. Calculate $\theta$:
   - Since $x = 0$, we need to handle this special case.
   - When $x = 0$ and $y < 0$, $\theta = \frac{3\pi}{2}$

Step 5. Calculate $\phi$ using the formula.

Step 6. Ensure $\theta$ is in the range $[0, 2\pi)$ and $\phi$ is in the range $[0, \pi]$.
</plan>
 
<code> 
from sympy import sqrt, atan2, acos, pi

def rectangular_to_spherical():
    x, y, z = 0, -3*sqrt(3), 3
    rho = sqrt(x**2 + y**2 + z**2)
    theta = atan2(y, x)
    phi = acos(z/rho)
    return rho, theta, phi

spherical_coordinates = rectangular_to_spherical()
print(spherical_coordinates)  
</code>

### Example Task 2 

<task>
Determine who lived longer between Lowell Sherman and Jonathan Kaplan.
</task>

<plan>
Step 1: Research the birth and death dates of Lowell Sherman.
Step 2: Research the birth and death dates of Jonathan Kaplan.
Step 3: Calculate the lifespan of each person in years.
Step 4: Compare the lifespans to determine who lived longer.
</plan>

<code>
from datetime import datetime

def calculate_lifespan(birth_date, death_date):
    birth = datetime.strptime(birth_date, "%Y-%m-%d")
    death = datetime.strptime(death_date, "%Y-%m-%d")
    return (death - birth).days / 365.25

def compare_lifespans():
    lowell_sherman = calculate_lifespan("1885-10-11", "1934-12-28")
    jonathan_kaplan = calculate_lifespan("1947-11-25", "2021-01-03")
    
    if lowell_sherman > jonathan_kaplan:
        return "Lowell Sherman"
    elif jonathan_kaplan > lowell_sherman:
        return "Jonathan Kaplan"
    else:
        return "They lived equally long"

result = compare_lifespans()
print(f"{result} lived longer.")
</code>


---

## Important Notes

Note that the above are some example tasks and output formats. You need to solve the current problem below.

---

## Current problem that we want to solve
<task>
<prompt> 
</task>

## Previous Solution

### Previous Reasoning Steps 
<plan>
<current_reasoning>
</plan>

### Previous Answer 
<final_answer>
<current_solution>
</final_answer>

--- 

## Critical Feedback 
<critical_feedback>

--- 

## Your Final Solution

Read the current problem in <task>...</task> again.

<task>
<prompt> 
</task>

To solve the current problem, you should first write the overall plan in <plan>...</plan> to solve the problem. Then, write python code in <code>...</code> tags to solve the problem.  If there is critical feedback and suggested plan, please revise your previous solution (if any) and provide the new plan and solution to solve the problem based on the critical feedback and suggested plan. 
Note that the code should not ask for any input from console, but it should be self-contained and print the final answer at the end.

## Remember to present your output in the following format:

<plan>
[Your general plan to solve the problem by using code. You can recall the required knowledge that you can use in the code, such as the facts, formulas, etc.]
</plan>

<code>
[Your python code to solve the current problem (instead of the example problems). Please print the final answer at the end of the code. Do not add any prefix like "The answer is" before the final answer]
</code> 
 
You must follow the format strictly, do not miss any field.  
Start your output by "<plan>...</plan>" and end your output by "<code> ... </code>".

