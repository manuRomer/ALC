# Test L06-metpot2k, Aval

import numpy as np
from Labo06.Labo06 import metpot2k, diagRH
from modulo_ALC import normaExacta

#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95


from modulo_ALC import *
import numpy as np
import time


# -------------------------------------------------------------

# =================================================================
# 1. Tests metpot2k (Matriz simétrica simple)
# =================================================================

# Inicialización para medición
total_time_mp2k_simple = 0
num_tests_mp2k_simple = 100

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
]).T

exitos = 0
for i in range(num_tests_mp2k_simple):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    
    # --- MEDICIÓN DE TIEMPO: INICIO ---
    start_time = time.time()
    v,l,_ = metpot2k(A,1e-15,1e5)
    end_time = time.time()
    # --- MEDICIÓN DE TIEMPO: FIN ---
    
    total_time_mp2k_simple += (end_time - start_time)
    
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95

# CÁLCULO DE PROMEDIO
avg_time_mp2k_simple = total_time_mp2k_simple / num_tests_mp2k_simple
print(f"✅ metpot2k (Simple):")
print(f"   Tiempo Total: {total_time_mp2k_simple:.6f} s")
print(f"   Tiempo Promedio por Test: {avg_time_mp2k_simple:.6f} s")

# =================================================================
# 2. Tests metpot2k con Householder (HH)
# =================================================================

# Inicialización para medición
total_time_mp2k_hh = 0
num_tests_mp2k_hh = 100

exitos = 0
for i in range(num_tests_mp2k_hh):
    v = np.random.rand(9)
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    
    # --- MEDICIÓN DE TIEMPO: INICIO ---
    start_time = time.time()
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    end_time = time.time()
    # --- MEDICIÓN DE TIEMPO: FIN ---
    
    total_time_mp2k_hh += (end_time - start_time)
    
    # max_eigen = abs(D[0][0]) (ya está ordenado por valor absoluto)
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95

# CÁLCULO DE PROMEDIO
avg_time_mp2k_hh = total_time_mp2k_hh / num_tests_mp2k_hh
print(f"\n✅ metpot2k (Householder):")
print(f"   Tiempo Total: {total_time_mp2k_hh:.6f} s")
print(f"   Tiempo Promedio por Test: {avg_time_mp2k_hh:.6f} s")

# =================================================================
# 3. Test diagRH (Matriz 3x3 simple)
# =================================================================

# Inicialización para medición
total_time_diagRH_simple = 0
num_tests_diagRH_simple = 1 # Es un solo test, pero lo tratamos como grupo

D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
]).T

A = S@D@S.T

# --- MEDICIÓN DE TIEMPO: INICIO ---
start_time = time.time()
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
end_time = time.time()
# --- MEDICIÓN DE TIEMPO: FIN ---

total_time_diagRH_simple = (end_time - start_time)

assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)

# CÁLCULO DE PROMEDIO (igual al total)
avg_time_diagRH_simple = total_time_diagRH_simple / num_tests_diagRH_simple
print(f"\n✅ diagRH (3x3 Determinístico):")
print(f"   Tiempo Total: {total_time_diagRH_simple:.6f} s")


# =================================================================
# 4. Tests diagRH (Matriz 5x5 aleatoria)
# =================================================================

# Inicialización para medición
total_time_diagRH_random = 0
num_tests_diagRH_random = 100

exitos = 0
for i in range(num_tests_diagRH_random):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    
    # --- MEDICIÓN DE TIEMPO: INICIO ---
    start_time = time.time()
    S,D = diagRH(A,tol=1e-15,K=1e5)
    end_time = time.time()
    # --- MEDICIÓN DE TIEMPO: FIN ---
    
    total_time_diagRH_random += (end_time - start_time)
    
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95

# CÁLCULO DE PROMEDIO
avg_time_diagRH_random = total_time_diagRH_random / num_tests_diagRH_random
print(f"\n✅ diagRH (5x5 Aleatorio):")
print(f"   Tiempo Total: {total_time_diagRH_random:.6f} s")
print(f"   Tiempo Promedio por Test: {avg_time_diagRH_random:.6f} s")

# =================================================================
# 5. Tiempo Total de Ejecución
# =================================================================

total_execution_time = (total_time_mp2k_simple + total_time_mp2k_hh + 
                        total_time_diagRH_simple + total_time_diagRH_random)
print(f"\n=============================================")
print(f"⏱️ TIEMPO TOTAL DE EJECUCIÓN DE TODOS LOS TESTS: {total_execution_time:.6f} s")
print(f"=============================================")