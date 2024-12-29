from sympy import *
import math
import numpy as np
x = Symbol('x')
y = x - exp(-x)
y_prime = y.diff(x)
print(y_prime)
y_prime_prime = y_prime.diff(x)
print(y_prime_prime)
y_prime = lambdify(x, y_prime, 'numpy')
v = y_prime(2)
print(v)