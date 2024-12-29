import numpy as np
np.set_printoptions(precision=100000000)

#Sprawdzenie czy macierz jest dodatnio określona
def is_positive_definite(matrix):
    n = matrix.shape[0]
    for i in range(1, n + 1):
        minor_matrix = matrix[:i, :i]
        if np.linalg.det(minor_matrix) <= 0:
            return False
    return True


def cholesky_banachiewicz(A, b):
    n = len(A)
    L = np.zeros((n, n))
    
    # Wyznaczenie macierzy L
    for i in range(n):
        for j in range(i+1):
            tmp_sum = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - tmp_sum))
    
    # Rozwiązywanie L*y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    
    # Rozwiązywanie L^T*x = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i+1, n))) / L[i][i]
    
    return x

# Macierz A i wektor b
A = np.array([[4, 8, -4], [8, 17, -1], [-4, -1, 57]])
b = np.array([-12, -17, 65])

print("\n", end="")
# Sprawdzenie, czy macierz A jest dodatnio określona
if is_positive_definite(A):
    print("Macierz A jest dodatnio okreslona.")
    # Rozwiązanie układu równań metodą Choleskiego-Banachiewicza
    x = cholesky_banachiewicz(A, b)
    print("Rozwiazanie ukladu rownan:           ", x)

    x_prec = np.linalg.solve(A, b)
    print("Rozwiązanie funkcją np.linalg.solve: ", x_prec)
else:
    print("Nie mozna zastosowac metody Choleskiego-Banachiewicza, poniewaz macierz A nie jest dodatnio okreslona.")

    x_prec = np.linalg.solve(A, b)
    print("Rozwiązanie funkcją np.linalg.solve: ", x_prec)



