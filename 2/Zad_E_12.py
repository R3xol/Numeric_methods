import numpy as np
import matplotlib.pyplot as plt

# Funkcja
def f(x):
    return x / (x**2 + 2)

# Wielomian aproksymacyjny - 2. stopnia
def approximation(x, a, b, c):
    return a * x**2 + b * x + c

# Przedział i krok
x_values = np.arange(-1.0, 1.01, 0.01)
y_values = f(x_values)

# Wyznaczenie współczynników wielomianu 2 stopnia
polynomial_coeffs = np.polyfit(x_values, y_values, 2)

# Obliczenie wartości aproksymacji wielomianowej
approx_values = approximation(x_values, *polynomial_coeffs)

# Błąd bezwzględny aproksymacji
absolute_error = np.abs(f(x_values) - approx_values)

# Błąd względny aproksymacji
relative_error = np.abs((f(x_values) - approx_values))/np.abs(f(x_values))*100

# Nie dzielimy przez 0
relative_error[100] = 'nan'

# Maksymalny błąd aproksymacji
max_error = np.max(absolute_error)

print("\n")
print(f"Maksymalny błąd aproksymacji: {max_error:.15f}")
print("\n")

# Maksymalny błąd względny aproksymacji
print("\n")
print(f"Maksymalny błąd względny: {relative_error[99]:.2f}%")
print("\n")

# Rysowanie wykresów
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Funkcja oryginalna')
plt.plot(x_values, approx_values, label='Aproksymacja')
plt.title('Funkcja oryginalna i jej aproksymacja wielomianem stopnia 2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wykres błędu aproksymacji
plt.figure(figsize=(10, 6))
plt.plot(x_values, absolute_error, label='Błąd aproksymacji', color='red')
plt.title('Bezwzględny błąd aproksymacji')
plt.xlabel('x')
plt.ylabel('Błąd bezwzględny')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wykres błędu aproksymacji
plt.figure(figsize=(10, 6))
plt.plot(x_values, relative_error, label='Błąd względny aproksymacji', color='red')
plt.title('Względny błąd aproksymacji')
plt.xlabel('x')
plt.ylabel('Błąd względny [%]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
