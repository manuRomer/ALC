#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    U = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    for k in range(0, n-1):
        for i in range(k+1, n):
            U[i][k] = U[i][k]/U[k][k]
            cant_op += 1
            for j in range(k+1, n):
                U[i][j] = U[i][j] - U[k][j] * U[i][k]
                cant_op += 1
    
    L = np.eye(n)
    for j in range(n):
        for i in range(j+1, n):
            L[i][j] = U[i][j]
            U[i][j] = 0
        
    valor = L@U

    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
