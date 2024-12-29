from email import header
import os
import cv2

import tifffile
from PIL import Image

from win32api import GetSystemMetrics
####
from ctypes import windll
windll.user32.SetProcessDpiAwarenessContext(-4)
#The -4 apparently signifies DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, which is the newest iteration of this type of context-awareness
####
import csv

import h5py

from scipy.signal import lfilter
from scipy.signal import medfilt

from cycler import cycler

import glob 

import gc
import math
import threading
from functools import partial

import matplotlib as mpl

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from imageio import imread
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import fftpack, io
from skimage.restoration import unwrap_phase

class Mesure():
    def __init__(self):
        self.how_many_files = 60
        self.z_vec = list(range(0, self.how_many_files, 1))
        
    def load_measurement(self):
        # wczytanie pliku hdf5    
        self.dir = "C:/Users/lyzwi/DHM/2023.06.14 ampicilina 2mg/timelapse-2023-06-14T15-18/2D_Plot_dn"
        name_of_file = "Data_and_settings.h5"
        # Filename
        filename = os.path.join(self.dir, name_of_file)

        files_h5 = []
        for i in os.listdir(self.dir):
            if os.path.isfile(os.path.join(self.dir,i)) and name_of_file in i:
                files_h5.append(i)

        is_h5_file = len(files_h5)

        print(is_h5_file)

        if is_h5_file != 0:
            h5 = h5py.File(filename,'r')

            self.loaded_measurement = h5['List_of_mesurments']
            self.loaded_measurement = list(self.loaded_measurement)

            self.ROI_reference = h5['ref_ROI_area']
            self.ROI_reference = np.matrix(self.ROI_reference)

            lam = h5['lambda']
            self.lam = lam[()]

            d_path = h5['d_path']
            self.d_path = d_path[()]

            name_substance = h5['name_substance']
            self.name_substance = name_substance.asstr()[()]

            alpha_v = h5['alpha_v']
            self.alpha_v = alpha_v[()]

            n0 = h5['n0']
            self.n0 = n0[()]

            pixel_size = h5['pixel_size']
            self.pixel_size = pixel_size[()]

            magnification = h5['magnification']
            self.magni = magnification[()]

            inverse_phase = h5['inverse_phase']
            inverse_phase = inverse_phase[()]
            self.inverse_phase = int(inverse_phase)

            hole_dia = h5['hole_dia']
            self.hole_dia = hole_dia[()]

            h5.close()

            print(self.lam)
            print(d_path)
            print(name_substance)

            self.loaded_images = self.loaded_measurement.copy()

            self.images_dn = self.loaded_images.copy()
            self.images_phase = self.loaded_images.copy()

            count = self.lam/(2*math.pi*self.d_path*10**6)

            number = 0
            for number in self.z_vec:
                self.image_dn = self.images_dn[number]
                self.images_dn[number] = self.image_dn*count
                number = number + 1
        print("Loaded measurement")

    def Concentration_dn(self):
    
        self.plot_image = self.images_dn.copy()
        image = self.plot_image[0]
        (self.hight, self.wighty) = image.shape[:2]
        self.h = list(range(0, self.hight, 1))

        mmppx=self.pixel_size*2*self.magni*10**(-3)

        vx = [i * mmppx for i in range(0, self.wighty)]
        vy = [i * mmppx for i in range(0, self.hight)]

        self.Concentration_dn_2D = np.zeros((self.hight,(int(10*self.how_many_files))))

        number = 0
        for number in self.z_vec:
            self.image = self.plot_image[number]
            self.mean_row = self.image.mean(axis=1)

            self.mean_row = self.mean_row - self.mean_row[int(self.mean_row.shape[0] / 2 + 70)]
            
            pom = number*10
            iter = list(range(pom, pom+10, 1))
            for n in iter:
                self.Concentration_dn_2D[:,n]=self.mean_row
            
            number = number + 1

        self.Concentration_dn_2D_poprawione = np.zeros((self.hight,(int(self.how_many_files))))

        number = 0
        for number in self.z_vec:
            self.image = self.plot_image[number]
            self.mean_row = self.image.mean(axis=1)

            self.mean_row = self.mean_row - self.mean_row[int(self.mean_row.shape[0] / 2 + 70)]
            
            self.Concentration_dn_2D_poprawione[:,number]=self.mean_row
            
            number = number + 1

        # wyznaczenie listy zawierajacej srednie kazdego wiersza
        self.Concentration_dn_1D = list(range(0, self.how_many_files, 1))

        number = 0
        for number in self.z_vec:
            self.image = self.plot_image[number]
            self.mean_row = self.image.mean(axis=1)
            self.mean_row = self.mean_row - self.mean_row[int(self.mean_row.shape[0] / 2 + 70)]
            self.Concentration_dn_1D[number]=self.mean_row
            number = number + 1


        print("2D Concentration")


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


self = Mesure()
self.load_measurement()
self.Concentration_dn()

mmppx=self.pixel_size*2*self.magni*10**(-3)
vy = [i * mmppx for i in range(0, self.hight)]

plt.style.use('classic')

plt.figure(figsize=(12,8))
fig = plt.gcf()
plt.plot(vy, self.Concentration_dn_1D[59])
plt.title("Wartość współczynnika załamania")
plt.xlabel("[mm]")
plt.ylabel("\u0394n")
plt.ylim(1.1*(np.min(self.Concentration_dn_1D[59])),1.1*(np.max(self.Concentration_dn_1D[59])))
plt.grid()
plt.show()

ref_roi_dn = self.ROI_reference * self.lam / (2 * np.pi * self.d_path * 10**6)

surf_ar = self.d_path * 10**(-6) * mmppx * 10**(-3)* np.size(self.Concentration_dn_1D[59], 0)
Nval = self.Concentration_dn_1D[59] * self.alpha_v * surf_ar

Nval = Nval[0:410]
vy = [i * mmppx for i in range(0, 410)]

plt.style.use('classic')

plt.figure(figsize=(12,8))
fig = plt.gcf()
plt.plot(vy, Nval)
plt.title("Stężenie molowe")
plt.xlabel("[mm]")
plt.ylabel("C [mol]")
plt.ylim(-0.1e-8,1.3*(np.max(Nval)))
plt.grid()
plt.show()

#######################################################################################################
#Część zwiżana z metodami numerycznymi
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Dopasowanie wielomianu
degree = 10 # Stopień wielomianu
coefficients = np.polyfit(vy, Nval, degree)
polynomial_fit = np.poly1d(coefficients)

# Wykres
plt.figure(figsize=(12,8))
plt.scatter(vy, Nval, color='blue', label='Dane', s=1)
plt.plot(vy, polynomial_fit(vy), color='red', label=f'Dopasowany wielomian (stopień {degree})')
plt.legend()
plt.xlabel("[mm]")
plt.ylabel("C [mol]")
plt.ylim(-0.1e-8,1.2*(np.max(Nval)))
plt.xlim(0,4)
plt.grid()
plt.show()

from scipy.integrate import quad

# Definiowanie funkcji wielomianowej
def polynomial_function(x):
    return polynomial_fit(x)

wielomian = []
blad_przybliżenia = []
for x in vy:
    wielomian.append(polynomial_function(x))

ss_res = np.sum((Nval - wielomian) ** 2)
ss_tot = np.sum((Nval - np.mean(Nval)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f'Wartość R^2: {r2}')

blad_bezwzględny_wielomian = np.abs(Nval - wielomian)
# Wykres
plt.figure(figsize=(12,8))
plt.plot(vy,blad_bezwzględny_wielomian)
plt.title("Błąd bezwzględny wielomianu przybliżającego")
plt.xlabel("[mm]")
plt.ylabel("Błąd bezwzględny")
plt.ylim(-0.1e-9,1.2*(np.max(blad_bezwzględny_wielomian)))
plt.grid()
plt.show()

blad_względny_wielomian = np.abs((Nval - wielomian) / Nval)
# Wykres
plt.figure(figsize=(12,8))
plt.plot(vy,blad_względny_wielomian)
plt.title("Błąd względny wielomianu przybliżającego")
plt.xlabel("[mm]")
plt.ylabel("Błąd bezwzględny")
plt.xlim(0,4)
plt.ylim(-0.1,1.2*(np.max(blad_względny_wielomian)))
plt.show()

# Policzenie całki
dokladna_wartosc, error = quad(polynomial_function, vy[0], vy[-1])

liczba_wezlow = np.arange(10, 1010, 10)

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

for n in liczba_wezlow:
    tab_wartosc.append(dokladna_wartosc)
    wynik_parabole, h_par = metoda_parabol(polynomial_function, vy[0], vy[-1], n)
    tab_parabole.append(wynik_parabole)
    wynik_trapezy, h_t = metoda_trapezow(polynomial_function, vy[0], vy[-1], n)
    tab_trapezy.append(wynik_trapezy)
    wynik_prostokaty, h_pro = metoda_prostokatow(polynomial_function, vy[0], vy[-1], n)
    tab_prostokaty.append(wynik_prostokaty)

    tab_h_p.append(h_t)

    blad_prostokaty.append(np.abs(dokladna_wartosc - wynik_prostokaty))
    blad_trapezy.append(np.abs(dokladna_wartosc - wynik_trapezy))
    blad_parabole.append(np.abs(dokladna_wartosc - wynik_parabole))

    blad_wzgledny_prostokaty.append((np.abs(dokladna_wartosc - wynik_prostokaty)/dokladna_wartosc)*100)
    blad_wzgledny_trapezy.append((np.abs(dokladna_wartosc - wynik_trapezy)/dokladna_wartosc)*100)
    blad_wzgledny_parabole.append((np.abs(dokladna_wartosc - wynik_parabole)/dokladna_wartosc)*100)


# Wyświetlenie wyniku
print(f"Stężenie substancji od x={vy[0]} do x={vy[-1]} wynosi: {dokladna_wartosc}")


# Wykres zależności błędu od liczby węzłów
plt.figure(figsize=(12, 8))
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
plt.figure(figsize=(12, 8))
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

print(f"Błąd względny metody prostokątów wynosi:  {blad_wzgledny_prostokaty[-1]}")
print(f"Błąd względny metody trapezów wynosi:     {blad_wzgledny_trapezy[-1]}")
print(f"Błąd względny metody parabol wynosi:      {blad_wzgledny_parabole[-1]}")