import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dane
xi = np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.3])
yi = np.array([750, 1000, 1400, 2000, 2700, 3750])

'''plt.figure(figsize=(10, 6))
plt.title('Dane wjściowe')
plt.scatter(xi, yi)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()'''

# Funkcja eksponencjalna
def exponential_f(x, a0, a1):
    return a0 * np.exp(a1 * x)

# Dopasowanie modelu eksponencjalnego metodą najmniejszych kwadratów
params_exp, covariance_exp = curve_fit(exponential_f, xi, yi)
a0_exp, a1_exp = params_exp

# Przygotowanie danych do regresji liniowej
x_lin = xi.reshape(-1, 1)
# Zastosowanie ln do przekształcenia danych
y_lin = np.log(yi)  

'''plt.figure(figsize=(10, 6))
plt.title('Zlinearyzowqane dane')
plt.scatter(x_lin, y_lin)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()'''

# Dopasowanie modelu liniowego
linear_regression = LinearRegression()
linear_regression.fit(x_lin, y_lin)

# Wyciągnięcie współczynników regresji liniowej
a1_lin = linear_regression.coef_[0]
a0_lin = np.exp(linear_regression.intercept_)

# Obliczenie przybliżeń punktów początkowych dla obu modeli
predict_exp_values = exponential_f(xi, a0_exp, a1_exp)
predict_lin_values = exponential_f(xi, a0_lin, a1_lin)

# Porównanie błędu R^2
r2_exp = r2_score(yi, predict_exp_values)
r2_lin = r2_score(yi, predict_lin_values)

print('\n',r2_exp-r2_lin)

print("\nParametry modelu eksponencjalnego:")
print("a0:", a0_exp)
print("a1:", a1_exp)

print("\nParametry modelu liniowego:")
print("a0:", a0_lin)
print("a1:", a1_lin)

print("\nWspółczynniki R^2:")
print("Model eksponencjalny:", r2_exp)
print("Model liniowy:       ", r2_lin)
print("\n")

# Funkcja modelu eksponencjalnego
def exponential_model(x_v, a0, a1):
    y_values = []
    for x in x_v:
        y_values.append(a0 * np.exp(a1 * x))
    return y_values

# Porównanie wyników na wykresie
plt.figure(figsize=(10, 6))
plt.scatter(xi, yi, label='Dane', s=15, c='black')
x_values = np.arange(0.3, 2.6, 0.1)
y_values_exp = exponential_model(x_values, a0_exp, a1_exp)
plt.plot(x_values,y_values_exp , label='Model eksponencjalny', color='red', linewidth=1)
y_values_lin = exponential_model(x_values, a0_lin, a1_lin)
plt.plot(x_values, y_values_lin, label='Regresja liniowa', color='green', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Porównanie dopasowania modelu eksponencjalnego i regresji liniowej')
plt.legend()
plt.grid(True)
plt.show()
