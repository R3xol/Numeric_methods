import math
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from scipy.optimize import root_scalar

np.set_printoptions(precision=100000)

# Wyznaczenie f'()
def f_prime_x():
    x = Symbol('x')
    y = x - exp(-x)
    y_prime = y.diff(x)
    y_prime = lambdify(x, y_prime, 'numpy')
    return y_prime

# Wyznaczenie f''()
def f_prime_prime_x():
    x = Symbol('x')
    y = x - exp(-x)
    y_prime = y.diff(x)
    y_prime_prime = y_prime.diff(x)
    y_prime_prime = lambdify(x, y_prime_prime, 'numpy')
    return y_prime_prime

# Wartość funkcji w punkcie x
def f(x):
    return x - math.exp(-x)

# Wartość pierwszej pochodnej w punkcie x
def f_prime(x):
    f_prime = f_prime_x()    
    return f_prime(x)

# Wartość drugiej pochodnej w punkcie x
def f_prime_prime(x):
    f_prime_prime = f_prime_prime_x()
    return f_prime_prime(x)


def secant_method(x0, x1, solution, precizion):
    iteration = 0
    tab_error = []
    tab_fx = []
    while True:
        # Wyznaczenie punktu przecięcia siecznej z osią OX
        x_new = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))

        tab_fx.append(abs(f(x_new)))

        error = abs(x_new - solution)
        if error != 0.0:
            tab_error.append(error)

        # Sprawdzenie dokładności przybliżenia
        if abs(f(x_new)) < precizion:
            return x_new, tab_error, tab_fx
        
        if f(x_new)*f(x0) > 0:
            x0 = x_new
        else:
            x1 = x_new
        iteration += 1
        

def newton_method(x0, x1, solution, precizion):
    # Jako pierwsze przybliżenie przyjmujemy koniec w którym
    # f(x) i f''(x) mają ten sam znak
    if f(x0)*f_prime_prime(x0)>=0:
        x0 = x0
    elif f(x1)*f_prime_prime(x1)>=0:
        x0 = x1
    else:
        print(":(")

    iteration = 0
    tab_error = []
    tab_fx = []
    while True:
        tab_fx.append(abs(f(x0)))
        # Wyznaczenie wartości błędu
        error = abs(abs(x0)-solution)
        if error != 0.0:
            tab_error.append(error)
        
        # Sprawdzenia warunku wyjścia
        if abs(f(x0)) < precizion or x1 < x0:
            return x0, tab_error, tab_fx

        x0 = x0 - f(x0) / f_prime(x0)

        iteration += 1

# Wyznaczenie rozwiązania równania x = exp(-x)
solution = root_scalar(f, bracket=[0, 2], method='brentq')
print("\n")
print("Rozwiązanie równania x = exp(-x):", solution.root)
print(end='')

solution_secant, Secant_error, Secant_fx = secant_method(0, 2, solution.root, 1e-12)
print(Secant_error)
solution_newton, Newton_error, Newton_fx = newton_method(0, 2, solution.root, 1e-6)

print("Rozwiązanie metodą siecznych:    ", solution_secant)
print("Rozwiązanie metodą Newtona:      ", solution_newton)
print("\n")

plt.figure(figsize=(10, 6))
plt.plot(Secant_error, label="Metoda siecznych")
plt.plot(Newton_error, label="Metoda Newtona")
plt.xlabel('Iteracja')
plt.ylabel('Błąd')
plt.yscale('log')
plt.legend()
plt.title('Błąd w stosunku do wartości wyznaczonej funkcją "root_scalar"')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Secant_fx, label="Metoda siecznych")
plt.plot(Newton_fx, label="Metoda Newtona")
plt.xlabel('Iteracja')
plt.ylabel('Błąd')
plt.yscale('log')
plt.legend()
plt.title('Wartość funkcji w punkcie dla kolejnych iteracji')
plt.show()