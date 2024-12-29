import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Zdefiniowanie funkcji do całkowania
def funkcja(x):
    return np.exp(-x**2)

# Metoda 3/8 Newtona
def metoda_trzy_osme(fun, a, b, npanel):
    h = (b - a) / npanel
    # x = np.linspace(a, b, npanel)
    integral = (fun(a) + fun(b))
    for i in range(1, npanel):
        x = 0.0
        x = a + i * h
        print(x)
        if i % 3 == 0:
            integral += 2 * fun(x)
        else:
            integral += 3 * fun(x)
    integral *= 3 * h / 8
    return integral, h

liczba_wezlow = [3, 6, 9, 12]
liczba_przedziałow = [(n - 1) * 3 for n in liczba_wezlow]

a = 0.0
b = 1.0

# Inicjalizacja list
blad_trzy_osme = []
blad_względny_trzy_osme = []
wynik_trzy_osme_tab = []
wynik_tab = []
tab_h = []

const = 2/np.sqrt(np.pi)

# Dokładna wartość całki
dokladna_wartosc, _ = quad(funkcja, a, b)
dokladna_wartosc *= const

# Obliczenia dla każdej liczby węzłów
for n in liczba_przedziałow:
    wynik_tab.append(dokladna_wartosc)
    #Metoda alternatywna (Trzeba iterować fora po liczba_wezlow )
    '''a = 0.0
    b = 1.0
    wezly = np.linspace(a, b, n)
    print(wezly)
    wynik_trzy_osme = 0.0
    for i in range(0,  np.size(wezly)-1):
        a_nowe = wezly[i]
        b_nowe = wezly[i+1]
        print(a_nowe, b_nowe)
        wynik_trzy_osme_i, h = metoda_trzy_osme(funkcja, a_nowe, b_nowe, 3)
        wynik_trzy_osme += wynik_trzy_osme_i
        tab_h.append(h)
    print('\n')'''
    wynik_trzy_osme, h = metoda_trzy_osme(funkcja, a, b, n)
    tab_h.append(h)

    wynik_trzy_osme *= const
    wynik_trzy_osme_tab.append(wynik_trzy_osme)
    blad_trzy_osme.append(np.abs(dokladna_wartosc - wynik_trzy_osme))
    blad_względny_trzy_osme.append((np.abs(dokladna_wartosc - wynik_trzy_osme)/dokladna_wartosc)*100)

print('\n')

dota_teams = ["liczba wezlow", "wartość całki", "metoda 3/8"]
data = [liczba_wezlow,
        wynik_tab,
        wynik_trzy_osme_tab]
format_row = "{:>22}" * (len(liczba_wezlow) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')

dota_teams = ["liczba wezlow", "blad bezwzgledny", "blad wzgledny [%]"]
data = [liczba_wezlow,
        blad_trzy_osme,
        blad_względny_trzy_osme]
format_row = "{:>25}" * (len(liczba_wezlow) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')

dota_teams = ["liczba wezlow", "liczba przedzaiłów", "długość przedziału"]
data = [liczba_wezlow,
        liczba_przedziałow,
        tab_h]
format_row = "{:>25}" * (len(liczba_wezlow) + 1)
for team, row in zip(dota_teams, data):
    print(format_row.format(team, *row))

print('\n')

# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(liczba_wezlow, blad_trzy_osme, label="Metoda 3/8 Newtona")
plt.xlabel("Liczba węzłów")
plt.ylabel("Błąd bezwzględny")
plt.title("Zależność błędu bezwzględnego od liczby węzłów dla metody 3/8 Newtona")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 6))
plt.plot(liczba_wezlow, blad_względny_trzy_osme, label="Metoda 3/8 Newtona")
plt.xlabel("Liczba węzłów")
plt.ylabel("Błąd względny")
plt.title("Zależność błędu względnego od liczby węzłów dla metody 3/8 Newtona")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()