import matplotlib.pyplot as plt
import numpy as np
import math

np.set_printoptions(precision=1000000)

# Przybliżenie sin(x) za pomocą szeregu Maclaurina do kolejno pierwszego, drugiego i trzeciego członu
def approx_sin_1(x):
    return x

def approx_sin_2(x):
    return x - (x**3)/math.factorial(3)

def approx_sin_3(x):
    return x - (x**3)/math.factorial(3) + (x**5)/math.factorial(5)


# Definicja zakresu
x = np.linspace(-0.3, 0.3, 1000)

# Wyliczenie wartości funkcji sin(x) oraz przryblirzeń szeregiem Maclaurina
sin_values = np.sin(x)
approx_1_values = approx_sin_1(x)
approx_2_values = approx_sin_2(x)
approx_3_values = approx_sin_3(x)

# Wyliczenie błędu względnego i bezwzględnego
absolute_errors_1 = np.abs(sin_values - approx_1_values)
relative_errors_1 = 100 * np.abs(absolute_errors_1 / sin_values)

absolute_errors_2 = np.abs(sin_values - approx_2_values)
relative_errors_2 = 100 * np.abs(absolute_errors_2 / sin_values)

absolute_errors_3 = np.abs(sin_values - approx_3_values)
relative_errors_3 = 100 * np.abs(absolute_errors_3 / sin_values)


# Wykres
plt.figure(figsize=(11, 5))

plt.subplot(2, 1, 1)
plt.plot(x, absolute_errors_1, label='1 czlon', linewidth=0.8)
plt.plot(x, absolute_errors_2, label='2 czlony', linewidth=0.8)
plt.plot(x, absolute_errors_3, label='3 czlony', linewidth=0.8)
plt.title('Blad bezwzgledny aproksymacji sin(x)')
plt.xlabel('x')
plt.ylabel('Blad bezwzgledny')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, relative_errors_1, label='1 czlon', linewidth=0.8)
plt.plot(x, relative_errors_2, label='2 czlony', linewidth=0.8)
plt.plot(x, relative_errors_3, label='3 czlony', linewidth=0.8)
plt.title('Blad wzgledny aproksymacji sin(x)')
plt.xlabel('x')
plt.ylabel('Blad wzgledny [%]')
plt.legend()

plt.tight_layout()
plt.show()