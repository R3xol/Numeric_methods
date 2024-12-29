import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Punkty interpolacji
survivor_points = [(9, 44.13), (20, 43.61), (30, 43.3), (40, 43.11), (50, 42.99), (60, 42.93)]

# Wyzanczenie funkcji bazowych Lagrange'a
def lagrange_base_f(i, x, points):
        x_i = [p[0] for p in points]
        return np.prod(x - np.delete(x_i, i)) / np.prod(x_i[i] - np.delete(x_i, i))

# Interpolacja Lagrange'a
def lagrange_interpolation(points):
    # Deklaracja przedziału dla którego wyznaczymy trasę Ledwolotusa
    X = np.linspace(min([p[0] for p in points])-3.1, max([p[0] for p in points])+3.1, 1000)
    Y = []

    # Wyznacznie wartości y dla poszczególnych x 
    #P(x) = L0(x)*y + L1(x)*y + L2(x)*y + L3(x)*y
    for x in X:
        y = 0
        for i, p in enumerate(points):
            y += p[1] * lagrange_base_f(i, x, points)
        Y.append(y)

    return X, Y

X, Y = lagrange_interpolation(survivor_points)



# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(X,Y)
plt.grid()
plt.xlabel('Promień [mm]')
plt.ylabel('Temperatura [°C]')
x_start = [points[0] for points in survivor_points]
y_start = [points[1] for points in survivor_points]
plt.scatter(x_start,y_start)
plt.show()