### Example Task 0

<task>
Let F be a constant unit force that is parallel to the vector (-1, 0, 1) in xyz-space. What is the work done by F on a particle that moves along the path given by (t, t^2, t^3) between time t=0 and time t=1?
What of the following is the right choice?
A. -1/4
B. -1/(4sqrt(2))
C. 0
D. sqrt(2)
</task>

<plan>
Step 1: Identify the Force and Path
    - The constant force \(\mathbf{F}\) is parallel to the vector \((-1, 0, 1)\), so we can write \(\mathbf{F} = F \cdot (-1, 0, 1)\). Since the force is a unit force, \(F = 1\), so \(\mathbf{F} = (-1, 0, 1)\).
    - The path is given by the position vector \(\mathbf{r}(t) = (t, t^2, t^3)\).

Step 2: Compute the Velocity
    - The velocity vector \(\mathbf{v}(t)\) is the time derivative of the position vector \(\mathbf{r}(t)\): \mathbf{v}(t) = \frac{d}{dt}(t, t^2, t^3) = (1, 2t, 3t^2)

Step 3: Compute the Dot Product of \(\mathbf{F}\) and \(\mathbf{v}(t)\)
    - Now, calculate the dot product \(\mathbf{F} \cdot \mathbf{v}(t)\):
    \mathbf{F} \cdot \mathbf{v}(t) = (-1, 0, 1) \cdot (1, 2t, 3t^2)
    \mathbf{F} \cdot \mathbf{v}(t) = -1 \cdot 1 + 0 \cdot 2t + 1 \cdot 3t^2 = -1 + 3t^2

Step 4: Set up the Integral for Work
    - Now, the work done is the integral of the dot product over the time interval from \(t = 0\) to \(t = 1\): W = \int_0^1 (-1 + 3t^2) \, dt

Step 5: Compute the Integral
    - We now compute the integral: W = \int_0^1 (-1 + 3t^2) \, dt = \left[ -t + t^3 \right]_0^1
    - Evaluating the integral: W = \left( -1 + 1^3 \right) - \left( 0 - 0^3 \right) = -1 + 1 = 0
</plan>

<code>
import sympy as sp

t = sp.symbols('t')
r = sp.Matrix([t, t**2, t**3])
v = r.diff(t)
F = sp.Matrix([-1, 0, 1])

dot_product = F.dot(v)
work = sp.integrate(dot_product, (t, 0, 1))

print(work)
</code>

### Example Task 1
 
<task>
Certain populations of finches have long been isolated on the Galapagos Islands off the western coast of South America. Compared with the larger stock population of mainland finches, these separate populations exhibit far greater variation over a wider range of species. The variation among these numerous finch species is the result of?
What of the following is the right choice?
A. convergent evolution
B. divergent evolution
C. disruptive selection
D. stabilizing selection
</task>

<plan>
Step 1: Understand the context of the question. The finches are isolated and have developed greater variation.
Step 2: Analyze the potential answers:
    - Convergent evolution is not the right answer because it involves unrelated species developing similar traits, not the variation caused by isolation.
    - Divergent evolution happens when populations become isolated and evolve different traits due to different environments or selective pressures.
    - Disruptive selection could lead to two distinct phenotypes, but it's a type of selection, not necessarily the cause of isolation-induced variation.
    - Stabilizing selection is not the right answer since it reduces variation rather than increasing it.
Step 3: After evaluating, (B) divergent evolution is the correct answer.
</plan>

<code>
def solve_finches_question():
    return "B. Divergent evolution"

answer = solve_finches_question()
print("The correct answer is:", answer)
</code>

### Example Task 2 

<task>
Determine who lived longer between Lowell Sherman and Jonathan Kaplan.
What of the following is the right choice?
A. Lowell lived longer
B. Jonathan lived longer
C. They lived the same longer
D. I don't know
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

