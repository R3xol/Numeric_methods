import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time

# Definicja funkcji
def funkcja(x):
    return x * np.exp(-x)

_x_ = np.arange(0, 5.01, 0.01)

# Wyznaczenie wartości funkcji y = x * exp(-x)
y = _x_ * np.exp(-_x_)

# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(_x_, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("funkcja x * np.exp(-x)")
plt.grid(True)
plt.show()

# Metoda prostokątów
def metoda_prostokatow(fun, a, b, npanel):
    h = (b - a) / (npanel)
    x = np.linspace(a, b, npanel+1)
    integral = 0
    # Sumowanie wartośći na środku przediału
    for i in range(0, npanel):
        integral += fun((x[i] + x[i+1]) / 2)
    # Przemożenie przez długość przediału
    integral = integral * h
    return integral, h

# Metoda trapezów
def metoda_trapezow(fun, a, b, npanel):
    h = (b - a) / (npanel)
    x = np.linspace(a, b, npanel+1)
    # Połowa wartości początu i końca przedziału
    integral = (fun(a) + fun(b)) / 2
    # Suma w węzłach
    for i in range(1, npanel):
        integral += fun(x[i])
    integral *= h
    return integral, h

# Metoda parabol
def metoda_parabol(fun, a, b, npanel):
    h = (b - a) / (npanel)
    x = np.linspace(a, b, npanel+1)
    integral = fun(a) + fun(b)
    for i in range(1, npanel):
        # Pażystą wartość razy 2 a niepażystą 4
        if i % 2 == 0:
            integral += 2 * fun(x[i])
        else:
            integral += 4 * fun(x[i])
    integral *= h / 3
    return integral, h

# Zakres całkowania
a = 0
b = 5

# Liczba węzłów do przetestowania
liczba_wezlow = [2, 4, 8, 16, 32]
liczba_przedziałow = [(n - 1) * 2 for n in liczba_wezlow]

print('\n')
print(liczba_przedziałow)

# Inicjalizacja list na wyniki
blad_prostokaty = []
blad_trapezy = []
blad_parabole = []

blad_wzgledny_prostokaty = []
blad_wzgledny_trapezy = []
blad_wzgledny_parabole = []

tab_h_p = []
tab_h_t = []
tab_h_pa = []

tab_wartosc = []
tab_prostokaty = []
tab_trapezy = []
tab_parabole =[]

dokladna_wartosc, _ = quad(funkcja, 0, 5)

# Obliczenia dla każdej liczby węzłów
for n in liczba_przedziałow:
    tab_wartosc.append(dokladna_wartosc)
    wynik_prostokaty, h_p = metoda_prostokatow(funkcja, a, b, n)
    tab_prostokaty.append(wynik_prostokaty)
    wynik_trapezy, h_t = metoda_trapezow(funkcja, a, b, n)
    tab_trapezy.append(wynik_trapezy)
    start_time = time.time()
    wynik_parabole, h_pa = metoda_parabol(funkcja, a, b, n)
    end_time = time.time()
    tab_parabole.append(wynik_parabole)

    tab_h_p.append(h_p)

    blad_prostokaty.append(np.abs(dokladna_wartosc - wynik_prostokaty))
    blad_trapezy.append(np.abs(dokladna_wartosc - wynik_trapezy))
    blad_parabole.append(np.abs(dokladna_wartosc - wynik_parabole))

    blad_wzgledny_prostokaty.append((np.abs(dokladna_wartosc - wynik_prostokaty)/dokladna_wartosc)*100)
    blad_wzgledny_trapezy.append((np.abs(dokladna_wartosc - wynik_trapezy)/dokladna_wartosc)*100)
    blad_wzgledny_parabole.append((np.abs(dokladna_wartosc - wynik_parabole)/dokladna_wartosc)*100)


print("\n")

dota_teams = ["liczba wezlow","wartosc dokladna", "metoda prostokatow", "metoda trapezow", "metoda parabol"]
data = [liczba_wezlow,
        tab_wartosc,
        tab_prostokaty,
        tab_trapezy,
        tab_parabole]

format_row = "{:>22}" * (len(liczba_wezlow) + 1)

for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print("\n")

dota_teams = ["liczba wezlow", "metoda prostokatow", "metoda trapezow", "metoda parabol"]
data = [liczba_wezlow,
        blad_prostokaty,
        blad_trapezy,
        blad_parabole]

format_row = "{:>25}" * (len(liczba_wezlow) + 1)

for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))


print("\n")

dota_teams = ["liczba wezlow", "metoda prostokatow", "metoda trapezow", "metoda parabol"]
data = [liczba_wezlow,
        blad_wzgledny_prostokaty,
        blad_wzgledny_trapezy,
        blad_wzgledny_parabole]

format_row = "{:>25}" * (len(liczba_wezlow) + 1)

for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))
    
print('\n')

dota_teams = ["liczba wezlow", "liczba przedziałów", "dlugosc przedialu"]
data = [liczba_wezlow,
        liczba_przedziałow,
        tab_h_p]
format_row = "{:>22}" * (len(liczba_wezlow) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')

# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(liczba_wezlow, blad_prostokaty, label="Metoda prostokątów")
plt.plot(liczba_wezlow, blad_trapezy, label="Metoda trapezów")
plt.plot(liczba_wezlow, blad_parabole, label="Metoda parabol")
plt.xlabel("Liczba węzłów")
plt.ylabel("Błąd bezwzględny")
plt.title("Zależność błędu od liczby węzłów dla różnych metod")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(liczba_wezlow, blad_wzgledny_prostokaty, label="Metoda prostokątów")
plt.plot(liczba_wezlow, blad_wzgledny_trapezy, label="Metoda trapezów")
plt.plot(liczba_wezlow, blad_wzgledny_parabole, label="Metoda parabol")
plt.xlabel("Liczba węzłów")
plt.ylabel("Błąd względny [%]")
plt.title("Zależność błędu względnego od liczby węzłów dla różnych metod")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
