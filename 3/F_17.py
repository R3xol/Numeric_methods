import numpy as np
import math
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

# Definicja funkcji podcałkowej
def funkcja(x):
    return x**2 * np.exp(-x**2)

def f(x):
    return x**2 

# Wartości wag i węzłów Gaussa-Hermite'a
wezly, wagi = hermgauss(5)

print(wezly,'\n')

def wielomian_hermita(n):
    # Funkcja tworząca wielomian Hermite'a danego stopnia
    if n == 0:
        return [1]
    elif n == 1:
        return [0, 1]
    else:
        h = np.zeros((8, 9))
        h[0, 0] = 1
        h[1, 1] = 2
        
        for i in range(2, n+1 ):
            macierz = np.copy(h[i - 1, :])
            for k in range(9 - 1, 0, -1):
                macierz[k] = macierz[k-1]
            macierz[0] = 0
            for j in range(9):
                h[i,j] = 2 * macierz[j] - 2 * (i-1) * h[i - 2, j]
      
        return h

def pochodna(wielomian_hermita_pochodna):
    for i in range(9):
        wielomian_hermita_pochodna[i] = wielomian_hermita_pochodna[i] * i

    for k in range(1, 8, 1):
        wielomian_hermita_pochodna[k-1] = wielomian_hermita_pochodna[k]
        wielomian_hermita_pochodna[8] = 0
    return wielomian_hermita_pochodna

h = wielomian_hermita(7)
n = 5

def print_matrix(matrix):
    print("\n")
    for i in range(9):
            print("x^", i, end="\t\t")
    print(end='\n')
    for row in matrix:
        for item in row:
            print(item, end="\t\t")
        print()
    print("\n")

print_matrix(h)

const = (2**(n+2)*math.factorial(n+1))

h_n_1 = np.copy(h[6, :])
h_n_1_pochodna = pochodna(h_n_1)

h_n_2 = np.copy(h[7, :])

h_1_pochodna = 0.0
h_2_ = 0.0

A_i = []

for x_i in wezly:
    for i in range(0, 7, 1):
        h_1_pochodna += h_n_1_pochodna[i]*(x_i**i)
        h_2_ += h_n_2[i] * x_i**i
    pom = const / (h_1_pochodna * h_2_)
    A_i.append(pom)

print('\n')
print(A_i)
print('\n')
print(wagi)
print('\n')

# Obliczenie wartości całki za pomocą kwadratury Gaussa-Hermite'a
wynik_gauss_hermite = np.sum(wagi * f(wezly))

# Obliczenie wartości całki za pomocą scipy.integrate.quad
wynik_scipy, _ = quad(funkcja, -np.inf, np.inf)

blad_bezwzgledny = (np.abs(wynik_scipy - wynik_gauss_hermite))
blad_wzgledny = ((np.abs(wynik_scipy - wynik_gauss_hermite)/wynik_scipy)*100)

# Wyświetlenie wyników
print("\n")
print("Wynik z kwadratury Gaussa-Hermite'a: ", wynik_gauss_hermite)
print("Wynik dokładny:                      ", wynik_scipy)

# Wyświetlenie wyników
print("\n")
print("Błąd bezwzględny:", blad_bezwzgledny)
print("Błąd względny:   ", blad_wzgledny)
print("\n")