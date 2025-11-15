import numpy as np
import math

## Laboratorio 1
tol = 1e-15

def error(x, y):
    """
    Recibe dos números x e y, y calcula el error de aproximar x usando y en float64.
    """
    return abs(x - y)

def error_relativo(x, y):
    """
    Recibe dos números x e y, y calcula el error relativo de aproximar x usando y en float64.
    """
    if abs(x) < tol:   # x muy cercano a 0
        return abs(y)
    return abs(x - y) / abs(x)

def matricesIguales(A, B, tol = 1e-8):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, además de distintos valores.
    """
    if A.shape != B.shape:
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if error_relativo(A[i, j], B[i, j]) > tol:
                return False
    return True

# Laboratorio 2

def rota(theta):
    """
    Recibe un ángulo theta y retorna una matriz de 2x2
    que rota un vector dado en un ángulo theta.
    """
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def escala(s):
    """
    Recibe una tira de números s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i].
    """
    n = len(s)
    res = np.zeros((n,n))
    for i in range (n):
        res[i][i] = s[i]
    return res

def rota_y_escala(theta, s):
    """
    Recibe un ángulo theta y una tira de números s,
    y retorna una matriz de 2x2 que rota el vector en un ángulo theta
    y luego lo escala en un factor s.
    """
    return multi_matricial(rota(theta), escala(s))

def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo mueve en un valor fijo b.
    """
    A = rota_y_escala(theta, s)
    T = np.eye(3)
    T[:2,:2] = A    # Inserto A en la parte superior izquierda
    T[:2,2] = b     # Inserto b en la última columna
    return T

def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un ángulo theta,
    una tira de números s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformación afín a v.
    """
    vh = np.array([v[0], v[1], 1])      # vector homogéneo
    wh = multi_matricial(afin(theta, s, b), traspuesta(vh))       # aplicar transformación
    return wh[:2]                       # volver a R2


## Laboratorio 3

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        return np.max(np.abs(x))
    else:
        return (np.sum(np.abs(x)**p))**(1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacíos, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    Y = X.copy()
    for i in range(len(Y)):
        vector = Y[i]
        n = norma(vector, p)
        if (n == 0):
            Y[i] = np.zeros_like(vector)
        else:
            Y[i] = (vector / n)
    return Y

def multi_matricial(A, B):

    # Si ambos son vectores 1D → producto interno
    if A.ndim == 1 and B.ndim == 1:
        if A.shape[0] != B.shape[0]:
            raise ValueError("Los vectores no tienen la misma longitud")
        return np.sum(A[i]*B[i] for i in range(A.shape[0]))

    # Si A es vector 1D → tratarlo como fila (1 x n)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    # Si B es vector 1D → tratarlo como col (n x 1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    # Chequear dimensiones compatibles
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"No se pueden multiplicar: {A.shape} y {B.shape}")

    # Inicializar matriz resultado con ceros
    C = np.zeros((A.shape[0], B.shape[1]))

    # Multiplicación clásica
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]

    # Si el resultado es un vector columna o fila → devolver como 1D
    
    if (C.ndim == 2 and C.shape[0] == 1 and C.shape[1] == 1):
        return C[0]
    if C.shape[1] == 1:
        return C.flatten()

    return C

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q, p} y el vector x en el cual se alcanza
    el máximo.
    """
    norma_max = 0; x_max = None
    for _ in range(Np):
        n = A.shape[1]
        x = 2*np.random.rand(n) - 1
        x_normalizado = x / norma(x, p)
        valor =  norma(multi_matricial(A, traspuesta(x_normalizado)), q)
        if valor > norma_max:
            norma_max = valor
            x_max = x_normalizado
    return norma_max, x_max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if p == 1:
        sumas_columnas = np.sum(np.abs(A), axis=0) 
        return np.max(sumas_columnas)
    if p == 'inf':
        sumas_filas = np.sum(np.abs(A), axis=1) 
        return np.max(sumas_filas)

def condMC(A, p, Np):
    """
    Devuelve el número de condición de A usando la norma inducida p.
    """
    inversa = np.linalg.inv(A)
    norma = normaMatMC(A,p,p,Np)
    normaInv = normaMatMC(inversa,p,p,Np)
    return norma[0] * normaInv[0]

def condExacta(A, p):
    """
    Devuelve el número de condición de A a partir de la fórmula de
    la ecuación (1) usando la norma p.
    """
    inversa = np.linalg.inv(A)
    return normaExacta(A,p) * normaExacta(inversa,p)
    return  

## Laboratorio 4

def calculaLU(A):
    """
    Calcula la factorizacion LU de la matriz A y retorna las matrices L
    y U, junto con el numero de operaciones realizadas. En caso de
    que la matriz no pueda factorizarse retorna None.
    """
    nops = 0
    m=A.shape[0]
    n=A.shape[1]
    U = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return None, None, 0
    for k in range(0, n-1):
        for i in range(k+1, n):
            if (U[k][k] == 0): return None, None, 0
            U[i][k] = U[i][k]/U[k][k]
            nops += 1
            for j in range(k+1, n):
                U[i][j] = U[i][j] - U[k][j] * U[i][k]
                nops += 2
    if (U[n-1][n-1] == 0): return None, None, 0
    L = np.eye(n)
    for j in range(n):
        for i in range(j+1, n):
            L[i][j] = U[i][j]
            U[i][j] = 0

    return L, U, nops

def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    inferior (por default asumir que es triangular inferior).
    """
    n = L.shape[1]
    if inferior:
        y = np.zeros(n)
        for i in range(0, n):
            y_i = b[i]
            for j in range(0, i):
                y_i -= L[i][j]*y[j]
            y[i] = y_i /  L[i][i]
        return y
    else:
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x_i = b[i]
            for j in range(n-1, i, -1):
                x_i -= L[i][j]*x[j]
            x[i] = x_i /  L[i][i]
        return x

def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares.
    """
    L, U, nops = calculaLU(A)
    if (L is None): return None
    n = L.shape[0]
    A_inv = np.zeros((n, n))
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1
        y = res_tri(L, e_j, inferior=True)   # L y e_j
        x = res_tri(U, y, inferior=False)    # U y y
        A_inv[:, j] = x                     # columna j de la inversa
    
    return A_inv

def traspuesta(A):
    if (A.ndim == 1):
        return A.reshape(-1, 1)

    n = A.shape[0]
    m = A.shape[1]
    At = np.zeros((m, n))
    for i in range (n):
        for j in range (m):
            At[j][i] = A[i][j]
    return At

def vector_traspuesto(v):
    n = v.shape[0]
    vt = np.zeros((n, 1))
    for i in range (n):
        vt[i][0] = v[i]
    return vt

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

def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    """
    L, U, nops1 = calculaLU(A)
    if (L is None): return None, None, None, 0
    Ut = traspuesta(U)
    V, D , nops2= calculaLU(Ut)

    return L, D, traspuesta(V) , nops1 + nops2

def esSDP(A, atol=1e-8):
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    """
    if (not simetrica(A)): return False

    L, D, V, nops = calculaLDV(A)
    if (D is None): return False
    for i in range(A.shape[0]):
        if (D[i][i] <= 0): return False
    return True

def producto_interno(v, w):
    res = 0
    for i in range(w.shape[0]):
        res += v[i]*w[i]
    return res

## Laboratorio 5

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de nxn
    tol la tolerancia con la que s efiltran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retrona matrices Q y R calculadas con Gram Schmidt (y como tercer argumento 
    opcional, el numero de operaciones)
    Si la matriz A no es de nxn, debe retornar None
    """
    nops = 0
    n = A.shape[1]
    a_1 = A[:, 0]
    r_11 = norma(a_1, 2)
    nops += a_1.shape[0]*2 -1     # operaciones de la norma
    
    q_1 = a_1 / r_11
    nops += 1
    
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    Q[:, 0] = q_1
    R[0][0] = r_11

    for j in range(1, n):
        q_j_prima = A[:, j]

        for k in range (0, j):
            q_k = Q[:, k]
            r_kj = producto_interno(q_k, q_j_prima)
            nops += q_k.shape[0]**2   # operaciones de la mulriplicacion matricial
            
            q_j_prima = q_j_prima - r_kj * q_k
            nops += 2 

            R[k][j] = r_kj
        
        r_jj = norma(q_j_prima, 2)
        nops += q_j_prima.shape[0]*2 -1     # operaciones de la norma
        
        q_j = q_j_prima / r_jj
        nops += 1 
        Q[:, j] = q_j
        R[j][j] = r_jj

    if (retorna_nops):
        return Q, R, nops
    return Q, R

def QR_con_HH(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de mxn (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m >= n, debe retornar None
    """
    m, n = A.shape
    if (m < n): return None, None

    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        x = R[k:, k]
        alpha = -np.sign(x[0])*norma(x, 2)
        e1 = np.zeros(x.shape[0])
        e1[0] = 1
        u = x- alpha * e1
        normaU = norma(u, 2)
        if (normaU > tol):
            u = u / normaU
            Hk = np.eye(x.shape[0]) - 2*multi_matricial(traspuesta(u), u)
            Hk_prima = np.eye(m)
            Hk_prima[k:, k:] = Hk
            R = multi_matricial(Hk_prima, R)
            Q = multi_matricial(Q, traspuesta(Hk_prima))
    
    return Q, R

def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de nxn
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectorers de Householder (RH) o GramSchmidt (GS)
    para realizar la factorizacion.
    retorna matrices Q y R calculadas con RH o GS (y como tercer argumento 
    opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if (metodo == 'GS'):
        return QR_con_GS(A, tol)
    else:
        return QR_con_HH(A, tol)

## Laboratorio 6

def f_A(A, v, k):
    w = v
    for i in range (k):
        w = multi_matricial(A, traspuesta(w))
        norma_2 = norma(w, 2)
        if (norma_2 <= 0): return np.zeros(v.shape[0]) 
        w = w/norma_2
    return w

def metpot2k(A, tol=1e-15, K=1000):
    """
    A una matriz de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del
    autovector.
    K el numero maximo de iteraciones a realizarse.
    Return vector v, autovalor lambda y numero de iteracion realizadas k
    """
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
    
    """
    A una matriz simetrica de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del
    autovector. K el numero maximo de iteraciones a realizarse
    return a matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T
    Si la matriz A no es simetrica, debe retornar None
    """
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)
    e1 = np.zeros(v1.shape[0])
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    w = v1 + sign * norma(v1, 2) * e1

    nw = norma(w, 2)
    if (nw < tol):
        H_v1 = np.eye(n)
    else:
        w = w / nw
        H_v1 = np.eye(n) - 2.0 * multi_matricial(traspuesta(w), w)

    if (n == 1):
        S = H_v1
        D = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        return S, D
    
    else:
        B = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        A_prima = B[1:,1:]
        S_prima, D_prima = diagRH(A_prima, tol, K)
        D = np.zeros((n, n))
        D[0][0] = lambda1
        D[1:,1:] = D_prima
        S = np.eye(n)
        S[1:,1:] = S_prima
        S = multi_matricial(H_v1, S)
        return S, D

## Laboratorio 7

def matriz_al_azar_valores_entre_0_y_1(n):
    A = np.zeros((n, n))
    for i in range (n):
        for j in range(n):
            A[i][j] = np.random.uniform(0, 1)
    return A

def normalizar_columnas(A, p):
    A = traspuesta(A)
    A = normaliza(A, p)
    return traspuesta(A)

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    if (n == 1):
        return np.eye(n)
    A = matriz_al_azar_valores_entre_0_y_1(n)
    return normalizar_columnas(A, 1)
    
def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    A = matriz_al_azar_valores_entre_0_y_1(n)
    for i in range(n):
        for j in range(n):
            if (A[i][j] <= thres):
                A[i][j] = 1
            else:
                A[i][j] = 0
    for i in range(n):
        if (norma(A[:, i], 1) == 0):
            A[np.random.randint(n)][i] = 1
    return normalizar_columnas(A, 1)
    
def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    A_prima = multi_matricial(traspuesta(A), A)
    S, D = diagRH(A_prima, tol, K=1e5)
    autovalores = np.diag(D)
    
    indices_nucleo = []
    for i in range(len(autovalores)):
        if np.abs(autovalores[i]) <= tol:
            indices_nucleo.append(i)
    if (indices_nucleo == []):
        return np.array([])
    base_nucleo = S[:, indices_nucleo]
    return base_nucleo

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    import numpy as np # Necesario para np.abs()

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, 
    lista con indices j, y lista con valores A_ij de la matriz A. 
    ...
    """
    dimensiones = (m_filas, n_columnas)

    if (listado == []):
        return {}, dimensiones
    
    indices_i = listado[0]
    indices_j = listado[1]
    valores_A_ij = listado[2]
    rala = {}
    num_elementos = len(indices_i)
    
    for k in range(num_elementos):
        
        valor = valores_A_ij[k]
        if np.abs(valor) >= tol:
            i = indices_i[k]
            j = indices_j[k]
            rala[(i, j)] = valor
            
    return rala, dimensiones

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    rala = A[0]
    m_filas, n_columnas = A[1]
    
    w = np.zeros(m_filas)
    
    for (i, j), valor_A_ij in rala.items():
        w[i] += valor_A_ij * v[j]
        
    return w

## Laboratorio 8

def svd_reducida(A,k="max",tol=1e-15):
    '''A la matriz de interes(de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k) '''
    
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


## Funciones necesarias para el Trabajo Practico

def cholesky(A):
    '''Devuelve la matriz L de cholesky'''
    if not(esSDP(A)):
        raise Exception("Para calcular la factorizacion de Cholesky de una matriz, es necesario que esta sea SDP")
    n = A.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        suma_diag = sum(L[j][k]**2 for k in range(j))
        L[j][j] = math.sqrt(A[j][j] - suma_diag)

        for i in range(j+1, n):
            suma_no_diag = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - suma_no_diag) / L[j][j]
    return L

def sustitucionParaAdelante(A, b):
    '''Resuelve un sistema lineal haciendo sustitucion de "arriba hacia abajo"'''
    x = np.zeros(A.shape[0])
    x[0] = b[0]/ A[0][0] 
    for i in range(1, A.shape[0]):
        sumatoria = 0
        for j in range(i):
            sumatoria = sumatoria + A[i][j]*x[j]
        x[i] = (b[i] - sumatoria) / A[i][i]
    
    return x

def sustitucionParaAtras(A, b):
    '''Resuelve un sistema lineal haciendo sustitucion de "abajo hacia arriba"'''
    x = np.zeros(A.shape[0])
    n = A.shape[0] -1
    x[n] = b[n] / A[n][n]
    
    for i in range(n-1, -1, -1):
        sumatoria = 0
        for j in range(i+1, n):
            sumatoria += A[i][j] * x[j]
        x[i] = (b[i] - sumatoria) / A[i][i]
    
    return x


def deVectorAMatrizInversa(VecEpsilon):
    '''dado un vector que representa una matriz diagonal, convierte el vector en matriz y calcula su inversa'''
    N = VecEpsilon.shape[0]
    matrizEpsilon = np.zeros((N, N))
    for i in range(N):
        matrizEpsilon[i][i] = 1/VecEpsilon[i]
    
    return matrizEpsilon
























