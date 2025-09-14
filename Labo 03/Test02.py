# tests_normaMC_plot.py
import numpy as np
import matplotlib.pyplot as plt
from Labo03 import normaMatMC, norma  # importa desde tu módulo principal

def run_tests(Np=1000, n_iter=100):
    # Defino matrices
    I = np.eye(2)
    A1 = np.array([[0, -1], [1, 0]])
    A2 = np.array([[1, 0], [0, 0]])
    A3 = np.array([[10, 10], [0, 0]])

    casos = [
        ('||I||2,1', I, 2, 1),
        ('||I||1,2', I, 1, 2),
        ('||I||2,inf', I, 2, 'inf'),
        ('||I||inf,2', I, 'inf', 2),
        ('||A1||2,2', A1, 2, 2),
        ('||A2||2,2', A2, 2, 2),
        ('||A2||inf,inf', A2, 'inf', 'inf'),
        ('||A3||2,inf', A3, 2, 'inf')
    ]

    plt.figure(figsize=(12, 8))
    
    for idx, (nombre, A, q, p) in enumerate(casos, 1):
        resultados = []
        for _ in range(n_iter):
            norma_est, _ = normaMatMC(A, q, p, Np)
            resultados.append(norma_est)
        resultados = np.array(resultados)

        # Imprimir estadísticas
        print(f"\nCaso: {nombre}")
        print(f"Valores estimados (min, max, media): {resultados.min():.4f}, {resultados.max():.4f}, {resultados.mean():.4f}")

        # Graficar histograma en subplot
        plt.subplot(2, 4, idx)
        plt.hist(resultados, bins=15, color='skyblue', edgecolor='black')
        plt.title(nombre)
        plt.xlabel('Norma estimada')
        plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tests()
