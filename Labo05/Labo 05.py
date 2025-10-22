import numpy as np
from ..modulo_ALC import *

def multi_traspuesta_por_vector(v, w):
    res = np.zeros(v.size)
    for i in range(v.size):
        res[i] = np.sum(v[i] * w)
    return res

def columna_j(A, j):
    n = A.shape[0]
    columna = np.zeros(n)
    for i in range(n):
        columna[i], A[i][j]
    return columna

def gramSchmidt(A, tol=1e-12):
    nops = 0
    n = A.shape[1]
    a_1 = columna_j(A, 0)
    r_11 = norma(a_1, 2)
    nops += a_1.size()*2 -1     # operaciones de la norma
    
    q_1 = a_1 / r_11
    nops += 1
    
    Q = np.zeros(n, n)
    R = np.zeros(n, n)
    Q[:, 0] = q_1
    R[1][1] = r_11

    r = 1
    columnas_ld = False
    for j in range(1, n):
        r = j
        if (columnas_ld): continue
        q_j_prima = columna_j(A, j)

        for k in range (0, j-1):
            q_k = columna_j(Q, k)
            r_kj = multi_matricial(traspuesta(q_k), q_j_prima)
            nops += q_k.size()**2   # operaciones de la mulriplicacion matricial
            
            q_j_prima = q_j_prima - r_kj * q_k
            nops += 2 

            R[k][j] = r_kj
        
        r_jj = norma(q_j_prima, 2)
        nops += q_j_prima.size()*2 -1     # operaciones de la norma
        
        if (r_jj > tol): 
            q_j = q_j_prima / r_jj
            nops += 1 
            Q[:, j] = q_j
            R[j][j] = r_jj
        else: columnas_ld

    if (columnas_ld):
        Q = Q[:, :r]
        R = R[:r, :]

    return Q, R, nops
    
