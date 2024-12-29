import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=100000000)

# Warunek dominanty przekątnej
def diagonal(matrix):
 
    n = len(matrix)
    result = []
 
    for i in range(n):
        diagonal = abs(matrix[i, i])  
        row_sum = np.sum(np.abs(matrix[i, :])) - diagonal
 
        result.append(diagonal > row_sum)
 
    return result

def jacobi(A, b, x, epsilon, max_iter):
    n = len(A)
    error_tab = []

    for k in range(max_iter):
        x_act = np.zeros(n)
        for i in range(n):
            x_act[i] = (b[i] - np.dot(A[i,:], x) + A[i,i]*x[i]) / A[i,i]
        
        error = np.linalg.norm(x_act - x)
        error_tab.append(error)
        
        if error < epsilon:
            break
        
        x = x_act.copy()
    
    return x, error_tab

def gauss_seidel(A, b, x, epsilon, max_iter):
    n = len(A)
    error_tab = []

    for k in range(max_iter):
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
        
        error = np.linalg.norm(np.dot(A, x) - b)
        error_tab.append(error)
        
        if error < epsilon:
            break
    
    return x, error_tab

# Testy dla równań a)
A = np.array([[4, -1, 1], [4, -8, 1], [-2, 1, 5]])
b = np.array([7, -21, 15])

# Testy dla równań b)
# A = np.array([[-2, 1, 5], [4, -8, 1], [4, -1, 1]])
# b = np.array([15, -21, 7])

print("\n", end="")
results = diagonal(A)

if all(results):
    print("Warunek dostateczny zbieznosci jest spelniony.")
    
    x0 = np.array([0.0, 0.0, 0.0])

    # Metoda Jacobiego
    x_jacobi_a, errors_jacobi_a = jacobi(A, b, x0, 1e-10, 100)
    print("Rozwiązanie równań metodą Jacobiego:      ", x_jacobi_a)

    # Metoda Gaussa-Seidela
    x0 = np.array([0.0, 0.0, 0.0])
    x_gauss_seidel_a, errors_gauss_seidel_a = gauss_seidel(A, b, x0, 1e-10, 100)
    print("Rozwiązanie równań metodą Gaussa-Seidela: ", x_gauss_seidel_a)

    x_prec = np.linalg.solve(A, b)
    print("Rozwiązanie funkcją np.linalg.solve:      ", x_prec)

    # Wykres
    plt.figure(figsize=(10, 5))

    plt.plot(range(len(errors_jacobi_a)), errors_jacobi_a, label='Metoda Jacobiego')
    plt.plot(range(len(errors_gauss_seidel_a)), errors_gauss_seidel_a, label='Metoda Gaussa-Seidela')
    plt.yscale('log')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Błąd')
    plt.title('Zbieżność metod iteracyjnych dla eps = 1e-10')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("Warunek dostateczny zbieznosci nie jest spelniony.")

