import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Funkcja interpolacji Newtona
def newton_coeffs(x, y):
    n = len(x)
    coeffs = y.copy()
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i-1]) / (x[i] - x[i-j])
            print(coeffs)
    return coeffs

# Wyznaczenie wartosci na podstawie wielomianu interpolacyjnego
def interpolated_temperature(t, times, coeffs):
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        pom = coeffs[i]
        for j in range(i):
            pom *= (t - times[j])
        result += pom
    return result

# Dane pomiarowe
times = [8, 9, 10, 11]              # Godziny pomiarów
temperatures = [20, 24, 26, 20]     # Temperatury dla poszczególnych godzin
print("\n")

# Obliczenie współczynników wielomianu interpolacyjnego
coeffs = newton_coeffs(times, temperatures)

# Interpolacja temperatury o godzinie 10:30
hour_1030 = 10.5
temperature_1030 = interpolated_temperature(hour_1030, times, coeffs)

print("\n")
print(f"Temperatura o godzinie 10:30: {temperature_1030:.2f} stopni Celsjusza")
print("\n")

X = np.linspace(7.5, 11.5, 25)
Y = []

def float_to_time(f_h):
    X_h=[]
    for float_hour in f_h:
        hour = int(float_hour)
        minute = int((float_hour - hour) * 60)
        X_h.append(f"{hour}:{minute:02d}")
    return X_h

X_hour = float_to_time(X)

for x in X:
    y = interpolated_temperature(x, times, coeffs)
    Y.append(y)

plt.figure(figsize=(16, 9))
plt.style.context('seaborn-white')
plt.plot(X_hour,Y)
plt.title("Wykres temperatury od czasu na egzaminie z Metod Numerycznych")
plt.xlabel('Godzina')
plt.ylabel('Temperatura (stopnie C)')
plt.scatter(float_to_time(times),temperatures)
plt.scatter("10:30" ,temperature_1030, color='red')
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.style.context('seaborn-white')
plt.plot(X,Y)
plt.title("Wykres temperatury od czasu na egzaminie z Metod Numerycznych")
plt.xlabel('Czas [h]')
plt.ylabel('Temperatura [stopnie C]')
plt.scatter(times,temperatures)
plt.grid()
plt.show()