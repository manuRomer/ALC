import numpy as np
from elim_gaussiana import *

def traspuesta(A):
    n = A.shape[0]
    m = A.shape[1]
    At = np.zeros((m, n))
    for i in range (n):
        for j in range (m):
            At[j][i] = A[i][j]
    return At

def simetrica(A):
    m=A.shape[0]
    n=A.shape[1]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    for i in range(n):
        for j in range(n):
            if (A[i][j] != A[j][i]):
                return False
    return True

def resolverLyb(L, b):
    n = L.shape[1]
    y = np.zeros(n)
    for i in range(0, n):
        y_i = b[i]
        for j in range(0, i):
            y_i -= L[i][j]*y[j]
        y[i] = y_i /  L[i][i]
    return y

def resolverUxy(U, y):
    n = U.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x_i = y[i]
        for j in range(n-1, i, -1):
            x_i -= U[i][j]*x[j]
        x[i] = x_i /  U[i][i]
    return x

def resolverAxbConLU(A, b):
    L, U, cant_op = elim_gaussiana(A)
    return resolverUxy(U, resolverLyb(L, b))

def calculaLDV(A):
    L, U = elim_gaussiana(A)
    Ut = traspuesta(U)
    D, V = elim_gaussiana(Ut)

    return L, D, V

def esSDP(A, atol=1e-10):
    if (not simetrica(A)): return False

    L, D, V = calculaLDV(A)
    for i in range(A.shape[0]):
        if (A[i][i] <= 0): return False
    return True

    
