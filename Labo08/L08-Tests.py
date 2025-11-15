import numpy as np
from modulo_ALC import multi_matricial, traspuesta, diagRH, matricesIguales
from math import sqrt
# Tests L08

def svd_reducida(A,k="max",tol=1e-15):
    m,n = A.shape
    
    #Optimizacion, si m < n calculamos primero U. 
    if m < n:
        A = traspuesta(A)
    
    U, diagonal_autovalores, V = calculoSVDReducida(A, tol)
    
    #Si m<n, se calculó primero U, que en la SVD de At toma el lugar de V (At = V Et Ut).
    #Swapeamos entonces para arreglar.
    if m < n:
        U, V = V, U        
    
    vector_epsilon = matriz_diagonal_a_vector(diagonal_autovalores)
    
    if not k == "max":
        U, vector_epsilon, V = retenerValoresSingulares(U, vector_epsilon, V, k)
    
    return U, vector_epsilon, V 

def retenerValoresSingulares(U, vector_epsilon, V, k):
    U_k = U[:, :k]
    V_k = V[:, :k]
    eps_k = vector_epsilon[:k]
    
    return U_k, eps_k, V_k

def calculoSVDReducida(A, tol):
    A_t_A = multi_matricial(traspuesta(A), A)
    
    V, diagonal_autovalores = diagRH(A_t_A)
    
    epsilon_hat, V_hat = reducirMatrices(V, diagonal_autovalores, tol)
    
    U_hat = calcularMatriz(A, V_hat, epsilon_hat)
    
    return U_hat, epsilon_hat, V_hat

def reducirMatrices(A, diagonal_autovalores, tol):
    for i in range(diagonal_autovalores.shape[0]):
        #if diagonal_autovalores[i][i] != 0
        if np.abs(diagonal_autovalores[i][i]) > tol: 
            diagonal_autovalores[i][i] = sqrt(diagonal_autovalores[i][i])
        else:
            epsilon_hat = diagonal_autovalores[:i, :i]
            A_hat = A[:,:i]
            return epsilon_hat, A_hat
    
    return diagonal_autovalores, A

def matriz_diagonal_a_vector(diagonal_autovalores):
    """Convierte una matriz diagonal a un array de numpy"""
    vector_epsilon = []
    
    for i in range(diagonal_autovalores.shape[0]):
        vector_epsilon.append(diagonal_autovalores[i][i])

    vector_epsilon = np.array(vector_epsilon)
    
    return vector_epsilon

def calcularMatriz(A,B,diagonal_autovalores):
    """Devuelve la matriz faltante para SVD"""
    matriz_faltante = multi_matricial(A, B) 
    for j in range(matriz_faltante.shape[1]):
        for i in range(matriz_faltante.shape[0]):
            matriz_faltante[i][j] = matriz_faltante[i][j] / diagonal_autovalores[j][j]
    return matriz_faltante



# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    
    if(not np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol)):
        print("Resultado")
        print(hU)
        print("Esperado")
        print(nU)
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    
    if(not np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol)):
        print("Resultado")
        print(hV)
        print("Esperado")
        print(nVT)
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

contador_test = 1
for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            # print("Se ejecuta la iteracion ", contador_test)
            contador_test = contador_test + 1
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo
contador_test = 1
m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        # print("Matrices con nucleo: Se ejecuta la iteracion ", contador_test)
        contador_test = contador_test + 1
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
contador_test = 1
A = np.random.random((8,6))
for k in [1,3,5]:
    # print("Tamaños de las matrices: Se ejecuta iteracion: ", contador_test)
    contador_test = contador_test + 1
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) == k, 'Tamaño de hS incorrecto'
