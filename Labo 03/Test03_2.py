
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
Np = 1000

# --- Test suite y gráficos ---
for i, A in enumerate(matrices):
    fig, axes = plt.subplots(len(normas), len(porcentajes), figsize=(15,10), sharex=False, sharey=False)
    fig.suptitle(f'Matriz {i+1}', fontsize=16)

    for row, p in enumerate(normas):
        for col, perc in enumerate(porcentajes):
            Ex, Eb, b_max, b_prima_max = dos_b_ii(A, p, perc, Np)
            r = np.array(Ex) / np.array(Eb)
            cond = np.linalg.cond(A, p=p if p != 'inf' else np.inf)

            ax = axes[row, col]
            ax.hist(r, bins=30, alpha=0.7)
            ax.axvline(x=cond, color='r', linestyle='--', label='cond(A)')
            ax.set_title(f'Norma={p}, perc={perc}%')
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()