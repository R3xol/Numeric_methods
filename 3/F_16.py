import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Definicja funkcji podcałkowej
def f(x):
    return x**2 

def funkcja(x):
    return x**2 / (np.sqrt((1 + x) * (1 - x)))

liczba_wezlow_tab = [11, 61, 101, 1001, 10001, 100001]
wynik_scipy, _ = quad(funkcja, -1, 1)

tab_scipy = []
tab_wartosci = []
tab_blad_wzgledny = []
tab_blad_bezwzgledny = []

for liczba_wezlow in liczba_wezlow_tab:
    tab_scipy.append(wynik_scipy)
    wezly = np.arange(1, liczba_wezlow+1, 1)
    wynik_gauss_czebyszew = 0.0
    pom = 0.0

    A_i = np.pi/(liczba_wezlow + 1)

    # Kwadratura Gaussa-Czebyszewa
    for i in wezly:
        x_i = np.cos(np.pi * ((2 * i )+ 1) / (2 * liczba_wezlow + 2))
        pom = (f(x_i ))
        wynik_gauss_czebyszew += pom

    wynik_gauss_czebyszew *= A_i
    tab_wartosci.append(wynik_gauss_czebyszew)
    tab_blad_bezwzgledny.append(np.abs(wynik_scipy - wynik_gauss_czebyszew))
    tab_blad_wzgledny.append((np.abs(wynik_scipy - wynik_gauss_czebyszew)/wynik_scipy)*100)

# Wyświetlenie wyników
print('\n')
print("Wynik z kwadratury Gaussa-Czebyszewa:", tab_wartosci[1])
print("Wynik dokładny:                      ", wynik_scipy)

print('\n')

dota_teams = ["liczba wezlow", "Wynik dokładny", "Gauss-Czebyszew"]
data = [liczba_wezlow_tab,
        tab_scipy,
        tab_wartosci]
format_row = "{:>25}" * (len(liczba_wezlow_tab) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')

dota_teams = ["liczba wezlow", "Błąd bezwzględny", "Błąd względny [%]"]
data = [liczba_wezlow_tab,
        tab_blad_bezwzgledny,
        tab_blad_wzgledny]
format_row = "{:>25}" * (len(liczba_wezlow_tab) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')


# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(liczba_wezlow_tab , tab_blad_wzgledny)
plt.xlabel("Liczba węzłów")
plt.ylabel("Błąd względny [%]")
plt.title("Zależność błędu względnego od liczby węzłów dla metody Gaussa-Czebyszewa")
plt.xscale('log')
plt.grid(True)
plt.show()