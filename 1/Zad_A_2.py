import numpy as np
import sys

eps_sys = sys.float_info.epsilon
print("Wartosc epsilon na podstawie funkcji (sys.float_info.epsilon):   ", end=" ")
print(eps_sys)


eps=1
# Dzielimy do czasu gdy suma eps i 1 daje 1 (eps jest poniżej precyzji)
while eps+1 != 1:
    eps /= 2
    i=eps+1
    if(i == 1):
        # Cofnięcie jednego kroku pętli ponieważ wartość jest poniżej dokładności
        eps = eps * 2
        break
    
print("Wartosc epsilon na algorytmu:                                    ", end=" ")
print(eps)

