import numpy as np

# Asumo que el input es de tipo np.ndarray para todas las funciones

def esCuadrada(A):
    if len(A) == 0:
        return True
    
    if not isinstance(A[0], (np.ndarray)):
       return False

    longestRowSize = len(max(A, key=len))
    shortestRowSize = len(min(A, key=len))
    return len(A) == longestRowSize == shortestRowSize
