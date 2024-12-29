import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.abs(x)

def runge_phenomenon(n_values):
    errors_equidistant = []
    errors_chebyshev = []

    for n in n_values:
        # Węzły równoodległe
        x_equidistant = np.linspace(-1, 1, n)
        y_equidistant = f(x_equidistant)
    
        # Interpolacja wielomianowa
        interpol_poly_e = np.polyfit(x_equidistant, y_equidistant, n - 1)
        x_values_e = np.linspace(-1, 1, 1000)
        y_values_e = np.polyval(interpol_poly_e, x_values_e)
        true_values_e = f(x_values_e)
        error_e = np.sum(np.abs(y_values_e - true_values_e))/1000
        errors_equidistant.append(error_e)

        # Węzły Czebyszewa
        x_chebyshev = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
        y_chebyshev = f(x_chebyshev)
        # Interpolacja wielomianowa
        interpol_poly = np.polyfit(x_chebyshev, y_chebyshev, n - 1)
        x_values = np.linspace(-1, 1, 1000)
        y_values = np.polyval(interpol_poly, x_values)
        true_values = f(x_values)
        error_c = np.sum(np.abs(y_values - true_values))/1000
        errors_chebyshev.append(error_c)

        '''# Rysowanie wykresu
        plt.figure(figsize=(16, 8))
        plt.subplot(1,2,1)
        plt.plot(x_equidistant, y_equidistant, color='red')
        plt.title("Węzły równoodległe")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(x_chebyshev, y_chebyshev)
        plt.title("Węzły Czebyszewa")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()'''

    return errors_equidistant, errors_chebyshev

n_values = np.arange(5, 21, 1)

errors_equidistant, errors_chebyshev = runge_phenomenon(n_values)

plt.figure(figsize=(10, 6))
plt.plot(n_values, errors_equidistant, label='Węzły równoodległe')
plt.plot(n_values, errors_chebyshev, label='Węzły Czebyszewa')
plt.xlabel('Liczba węzłów (n)')
plt.ylabel('Błąd bezwzględny')
plt.title('Błąd Rungego')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()