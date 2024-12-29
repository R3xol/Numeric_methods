import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.metrics import r2_score

# Definicja funkcji
def fun(x):
    return 2 + 2 * np.sin(x)**2 + np.cos(x)**2

# Symboliczna definicja funkcji dla obliczenia pochodnej analitycznej
x = sp.symbols('x')
y = 2 + 2 * sp.sin(x)**2 + sp.cos(x)**2

# Obliczenie pochodnej
dy_analitycznie = sp.diff(y, x)
print(dy_analitycznie)

# Konwersja pochodnej analitycznej na funkcję
dy_analitycznie_func = sp.lambdify(x, dy_analitycznie, 'numpy')

# Numeryczne obliczanie pochodnej
def numeryczna_pochodna(f, x, h):
    return (f(x + h) - f(x)) / h

# Zakres i krok różniczkowania
x_values = np.arange(0.0, 5.01, 0.01)
h = 0.01

# Obliczenie numerycznej pochodnej
dy_numerycznie = numeryczna_pochodna(fun, x_values, h)

dy_analitycznie_y = dy_analitycznie_func(x_values)

blad_bezwzględny = np.abs(dy_numerycznie - dy_analitycznie_y)

r2 = r2_score(dy_numerycznie, dy_analitycznie_y)

print("\n")
max = np.max(blad_bezwzględny)
print("Maksymalna wartość błędu bezwzględnego:",max)

min = np.min(blad_bezwzględny)
print("Minimalna wartość błędu bezwzględnego:", min)
print("\n")

# Obliczanie błędu kwadratowego dla każdej pary wartości
błędy_kwadratowe = blad_bezwzględny ** 2

n = 5.0 / 0.01 -1

# Obliczanie błędu średniokwadratowego
RMS = np.sqrt(np.sum(błędy_kwadratowe)/(n-1))

print("Błąd średniokwadratowy:", RMS)

print("\n")

print("Wartość współczynnika R^2:", r2)

print("\n")

# Wykres wyników
plt.figure(figsize=(12, 6))
plt.plot(x_values, dy_numerycznie, label='Numerycznie')
plt.plot(x_values, dy_analitycznie_y, label='Analitycznie', linestyle='--')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.title('Porównanie numerycznej i analitycznej pochodnej')
plt.legend()
plt.grid(True)
plt.show()

# Wykres wyników
plt.figure(figsize=(12, 6))
plt.plot(x_values, blad_bezwzględny)
plt.xlabel('x')
plt.ylabel('Błąd bezwzględny')
plt.title('Błąd bezwzględny dla poszczególnych wartości z przedziału [0, 5]')
plt.yscale('log')
plt.grid(True)
plt.show()
