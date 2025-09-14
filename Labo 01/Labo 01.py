import numpy as np
import math
import matplotlib.pyplot as plt

def salto():
    print('\n')

## Ejercicio 1

print(0.3+0.25)
print(0.3-0.25)

salto()

## Ejercicio 2

print('a) \n')
print(np.sqrt(2)**2-2)

salto()
print('b) \n')
valores = np.linspace(0.5, 1e-8, 100)

def funcion_a(x):
    return np.sqrt(2*(x**2)+1)-1

def funcion_b(x):
    return (2*x**2) / (np.sqrt(2*(x**2)+1)+1)

valores_usando_a = list(map(funcion_a, valores))
valores_usando_b = list(map(funcion_b, valores))

resultado = list(zip(valores_usando_a, valores_usando_b))


# Graficar ambas en la misma figura para comparar
plt.plot(valores, valores_usando_a, label="fórmula a")
plt.plot(valores, valores_usando_b, label="fórmula b", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de pérdida de significancia")
plt.legend()
plt.grid(True)
plt.show()


## Ejercicio 3

def ej3(n):
    if n == 1:
        return np.sqrt(2)
    else:
        return ej3(n-1)**2/np.sqrt(2)
    

def ej3():
    lista = []
    lista.append(np.sqrt(2))
    for i in range (1, 100):
        x_n = lista[i-1]
        lista.append(x_n**2/np.sqrt(2))
    return lista

plt.plot(ej3())
plt.ylim(1.414, 1.415)
plt.show()

## Ejercicio 4

def sumatoria_1_32b(n):
    s = np.float32(0)
    for i in range(1,10**n+1):
        s = s + np.float32(1/i)
    return s

def sumatoria_2_32b(n):
    s = np.float32(0)
    for i in range(1,5*10**n+1):
        s = s + np.float32(1/i)
    return s

def sumatoria_1_64b(n):
    s = np.float64(0)
    for i in range(1,10**n+1):
        s = s + np.float64(1/i)
    return s

def sumatoria_2_64b(n):
    s = np.float64(0)
    for i in range(1,5*10**n+1):
        s = s + np.float64(1/i)
    return s

print("Sumatoria 1 con n = 6: ")
print((sumatoria_1_32b(6), sumatoria_1_64b(6)))

print("Sumatoria 1 con n = 7: ")
print((sumatoria_1_32b(7), sumatoria_1_64b(7)))

print("Sumatoria 2 con n = 6: ")
print((sumatoria_2_32b(6), sumatoria_2_64b(6)))

print("Sumatoria 2 con n = 7: ")
print((sumatoria_2_32b(7), sumatoria_2_64b(7)))

# Lo que cambia aca es que se arranca sumando los numeros chiquitos lo cual evita tener que 
# sumarle un numero muy chiquito a un numero relativamente gigante a ese 

s = np.float32(0)
for i in range(5*10**7,0,-1):
    s = s + np.float32(1/i)
print("suma = ", s)

e = np.float32(0)
for i in range(10,-1,-1):
    e = e + np.float32(1/math.factorial(i))
print("suma = ", e)
print(np.e)

## Ejercicio 5

def son_iguales(A, B):
    if A.shape != B.shape:
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != B[i, j]:
                return False
    return True

A = np.array([
    [4.0, 2.0, 1.0],
    [2.0, 7.0, 9.0],
    [0.0, 5.0, 22.0/3.0]
])

L = np.array([
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 5.0/6.0, 1.0]
])

U = np.array([
    [4.0, 2.0, 1.0],
    [0.0, 6.0, 8.5],
    [0.0, 0.0, 0.25]
])


print(son_iguales(A,L@U))

