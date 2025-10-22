import numpy as np
from modulo_ALC import *

def f_A(A, v, k):
    w = v
    for i in range (k):
        w = multi_matricial(A, traspuesta(w))
        norma_2 = norma(w, 2)
        if (norma_2 <= 0): return np.zeros(v.shape[0]) 
        w = w/norma_2
    return w

def metpot2k(A, tol=1e-15, K=1000):
    v = np.random.rand(A.shape[0])
    v_prima = f_A(A, v, 2)
    e = multi_matricial(v_prima, traspuesta(v))
    k = 0
    while (abs(e-1)>tol and k < K):
        v = v_prima
        v_prima = f_A(A, v, 1)
        e = multi_matricial(v_prima, traspuesta(v))
        k += 1
    autovalor = multi_matricial(v_prima, multi_matricial(A, traspuesta(v_prima)))
    e -= 1
    return v, autovalor, k

def diagRH(A,tol=1e-15,K=1000):
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)
    e1 = np.zeros(v1.shape[0])
    e1[0] = 1
    w = e1 - v1
    H_v1 = np.eye(n) - 2 * (multi_matricial(traspuesta(w), w) / (norma(w, 2) ** 2))
    if (n == 2):
        S = H_v1
        D = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        return S, D
    else:
        B = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        A_prima = B[1:,1:]
        S_prima, D_prima = diagRH(A_prima, tol, K)
        D = np.eye(n)
        D[0][0] = lambda1
        D[1:,1:] = D_prima
        S = np.eye(n)
        S[1:,1:] = S_prima
        S = multi_matricial(H_v1, S)
        return S, D