# Zamiana liczb z systemu dziesiętnego na szestnastkowy
def dec_to_hex(dec_number):
    while dec_number > 0:
        quotient = dec_number % 16
        if quotient == 10:
            hex_number.append("A")
        elif quotient == 11:
            hex_number.append("B")
        elif quotient == 12:
            hex_number.append("C")
        elif quotient == 13:
            hex_number.append("D")
        elif quotient == 14:
            hex_number.append("E")
        elif quotient == 15:
            hex_number.append("F")
        else:
            hex_number.append(quotient)
        dec_number = dec_number // 16
    hex_number.reverse()
    return hex_number 

# Zamiana liczb z systemu dziesiętnego na ósemkowy
def dec_to_oct(dec_number):
    while dec_number > 0:
        quotient = dec_number % 8
        oct_number.append(quotient)
        dec_number //= 8
    oct_number.reverse()
    return oct_number

oct_number = []
hex_number = []

number = 2044

truth_h = hex(number)
truth_o = oct(number)

number_hex = dec_to_hex(number)
number_oct = dec_to_oct(number)

print("Wprowadzona liczba w systemie dziesietnym:               ", number)

print("Na podstaei algorytmu (liczba w systemie szesnastkowym): ", end=" ")
for var in number_hex:
    print(var, end="",)
print("\n", end="")
print("Na podstaei funkcji Python hex:                          " ,truth_h)

print("Na podstaei algorytmu (liczba w systemie osemkowym):     ", end=" ")
for var in number_oct:
    print(var, end="")
print("\n", end="")
print("Na podstaei funkcji Python oct:                          ", truth_o)