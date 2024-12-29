import numpy as np

np.set_printoptions(precision=1000000)

# Wyznaczenie norm macierzowych
def calculate_condition_numbers(A):
    cond_1 = np.linalg.cond(A, 1)           # Norma kolumnowa
    cond_2 = np.linalg.cond(A, 2)           # Norma spektralna
    cond_Inf = np.linalg.cond(A, np.inf)    # Norma nieskończoności
    return cond_1, cond_2, cond_Inf

# Sprawdza czy wszystkie minory macierzy są niezerowe
# Zwraca True jeśli wszystkie minory są niezerowe, False w przeciwnym razie.
def are_all_minors_non_zero(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            minor_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            minor_det = np.linalg.det(minor_matrix)
            if minor_det == 0:
                return False
    return True

# Zamiana wiersza pierwszego na wiersz bez zerowych elementów
def zamien_wiersze(macierz):
    cnt = 0
    # Sprawdzenie, czy istnieje zero w pierwszym wierszu
    if 0 in macierz[0]:
        # Szukanie wierszy z zerem
        for i in range(1, len(macierz)):
            if 0 not in macierz[i]:
                # Znaleziono wiersz bez zera, zamiana miejscami
                pom = macierz[0].copy()
                macierz[0], macierz[i] = macierz[i], pom
            else:  
                break
    
    return macierz

# Metoda elimincaji Gaussa
def gauss(A, b):
    n = len(b)
    Ab = np.column_stack((A, b))
    x = np.zeros(n)

    # Sprawdzenie warunku podstawowego
    is_posible = are_all_minors_non_zero(A)

    if is_posible == True:
        # Ustawienie w pierwszym wierszu, wiersza bez zer
        Ab = zamien_wiersze(Ab)
        print("Macierz wejściowa:       \n", end="")
        print(Ab)
        for i in range(n):
            # Odejmowanie wierszy
            for j in range(i+1, n):
                ratio = Ab[j, i] / Ab[i, i]
                Ab[j] =  Ab[j] - ratio * Ab[i]
        print("Macierz po wyzerowaniu:  \n", end="")
        print(Ab)
        # rozwiązanie
        for i in range(n-1, -1, -1):
            x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

        return x, True
    else:
        return x, False


# Układ równań
#(a)
#A = np.array([[1, 2], [1.1, 2]])
#b = np.array([10, 10.4])
#(b)
#A = np.array([[2, 5.999999], [2, 6]])
#b = np.array([8.000001, 8])
#(c)
A = np.array([[5.0, 3.0, 4.0], [3.0, 0.0, 1.0], [6.0, 3.0, 6.0]])
b = np.array([18.0, 7.0, 27.0])

print("\n", end="")
cond_1, cond_2, cond_Inf = calculate_condition_numbers(A)
print("Wskaźniki uwarunkowania:     ")
print("Norma kolumnowa:             ", cond_1)
print("Norma spektralna:            ", cond_2)
print("Norma nieskończoności:       ", cond_Inf)

print("\n", end="")
x, is_posible = gauss(A, b)
if is_posible == True:
    print("Rozwiązanie układu równań:           ", x)
    x_prec = np.linalg.solve(A, b)
    print("Rozwiązanie funkcją np.linalg.solve: ", x_prec)
    difrence = x_prec - x
    print("\n", end="")
    print("Roznica wynosi                     : ", difrence)
    print("\n", end="")
else:
    print("Nie spełniono warunku podstawowego")



