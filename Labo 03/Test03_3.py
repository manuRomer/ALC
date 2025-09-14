import numpy as np
import matplotlib.pyplot as plt
from Labo03 import normaMatMC, norma, dos_b_ii, variaPerc, condMC  # importa desde tu módulo principal

# --- Definición de matrices, normas y porcentajes ---
matrices = [
    np.array([[1, -1], 
              [1, 1]]),

    np.array([[501, 499], 
              [500, 500]]),

    np.array([[1000, 1/1000], 
              [0, 1000]]),
              
    np.array([[1/1000, 1000], 
              [0, 1000]])
]

normas = [1, 2, 'inf']
porcentajes = [1, 5, 10]
Np = 1000000
n_busquedas = 10


for A_idx, A in enumerate(matrices, 1):
    for p in normas:
        for perc in porcentajes:
            plt.figure(figsize=(6,6))
            plt.title(f"Matriz {A_idx}, norma p={p}, perc={perc}%")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().set_aspect('equal')

            for _ in range(n_busquedas):
                Ex, Eb, b_max, b_prima_max = dos_b_ii(A, p, perc, Np)
                delta_b = b_prima_max - b_max

                # Graficar b_max como origen y delta_b como flecha
                plt.arrow(0, 0, b_max[0], b_max[1], head_width=0.05, color='blue', label='b')
                plt.arrow(b_max[0], b_max[1], delta_b[0], delta_b[1], head_width=0.05, color='red', label='Δb')

            plt.grid(True)
            plt.show()