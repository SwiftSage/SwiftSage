### Example Task 0

<task>
Let $m=\underbrace{22222222}_{\text{8 digits}}$ and $n=\underbrace{444444444}_{\text{9 digits}}$.

What is $\gcd(m,n)$?
</task>

<plan>
Step 1. Identify the numbers for which we need to find the GCD: Let \( m = 22222222 \) and \( n = 444444444 \).

Step 2. Recall the Euclidean algorithm for finding the GCD, which states that \( \gcd(a, b) = \gcd(b, a \mod b) \).

Step 3. Apply the Euclidean algorithm iteratively:
   - Set \( a = n = 444444444 \) and \( b = m = 22222222 \).
   - Compute \( a \mod b \).

Step 4. Repeat the process until \( b = 0 \):
   - If \( b \) becomes zero, then the GCD is the value of \( a \).

Step 5. Return the result of the GCD.
</plan>
 
<code>
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

m = 22222222
n = 444444444

gcd_result = gcd(m, n)
print(gcd_result)
</code>

### Example Task 1
 
<task>
Find the minimum of the function
\[\frac{xy}{x^2 + y^2}\]in the domain $\frac{2}{5} \le x \le \frac{1}{2}$ and $\frac{1}{3} \le y \le \frac{3}{8}.$
</task>

<plan>
Step 1. Identify the function to minimize: 
   \[\frac{xy}{x^2 + y^2}\]

Step 2. Note the constraints of the domain: 
   \[\frac{2}{5} \le x \le \frac{1}{2}, \quad \frac{1}{3} \le y \le \frac{3}{8}.\]

Step 3. Recognize that the function is not differentiable at $(0, 0)$, thus we need to consider the boundaries of the domain.

Step 4. Evaluate the function at the corners of the rectangular boundary defined by the limits of $x$ and $y$.

Step 5. Compare the function values obtained from boundary evaluations to find the minimum.
</plan>
 
<code> 
import numpy as np

# Define the function
def f(x, y):
    return x * y / (x**2 + y**2)

# Define the domain
x = np.linspace(2/5, 1/2, 100)
y = np.linspace(1/3, 3/8, 100)
X, Y = np.meshgrid(x, y)

# Calculate the function values on the domain
Z = f(X, Y)

# Find the minimum value
min_value = np.min(Z)
print(min_value)
</code>


### Example Task 2 

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

