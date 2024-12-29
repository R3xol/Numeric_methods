import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Punkty interpolacji
survivor_points = [(0, -4), (1, 3), (3, 5), (2, 2)]

# Wyzanczenie funkcji bazowych Lagrange'a
def lagrange_base_f(i, x, points):
        x_i = [p[0] for p in points]
        return np.prod(x - np.delete(x_i, i)) / np.prod(x_i[i] - np.delete(x_i, i))

# Interpolacja Lagrange'a
def lagrange_interpolation(points):
    # Deklaracja przedziału dla którego wyznaczymy trasę Ledwolotusa
    X = np.linspace(min([p[0] for p in points])-0.1, max([p[0] for p in points])+0.1, 1000)
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
plt.title('Trasa lotu Ledwolotusa')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
x_start = [points[0] for points in survivor_points]
y_start = [points[1] for points in survivor_points]
plt.scatter(x_start,y_start)
plt.show()