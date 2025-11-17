import numpy as np

def QR_con_GS_optimizado(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de nxn
    tol la tolerancia con la que s efiltran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retrona matrices Q y R calculadas con Gram Schmidt (y como tercer argumento 
    opcional, el numero de operaciones)
    Si la matriz A no es de nxn, debe retornar None
    """
    nops = 0
    m, n = A.shape
    a_1 = A[:, 0]
    r_11 = norma(a_1, 2)
    nops += a_1.shape[0]*2 -1     # operaciones de la norma
    
    q_1 = a_1 / r_11
    nops += 1
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    Q[:, 0] = q_1
    R[0][0] = r_11

    for j in range(1, n):
        q_j_prima = A[:, j] 
        
        # OPTIMIZACION: vectorizar el calculo de todos los coeficientes r_{k,j} para una columna j
        R_col_j = Q[:, :j].T @ q_j_prima 
        # OPTIMIZACION: vectorizar la resta de las proyecciones
        proyeccion = Q[:, :j] @ R_col_j 
        q_j_prima = q_j_prima - proyeccion
        R[:j, j] = R_col_j 
        
        r_jj = norma(q_j_prima, 2)
        
        q_j = q_j_prima / r_jj
        
        Q[:, j] = q_j
        R[j, j] = r_jj
    

    R_economica = R[:n, :] 
    Q_economica = Q[:, :n]

    if (retorna_nops):
        return Q_economica, R_economica, nops

    
    return Q_economica, R_economica

def QR_con_HH_optimizado(A, tol=1e-12, retorna_nops=False):
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
        if normaU > tol:
            u /= normaU

            # OPTIMIZACION: plicar la transformación de Householder directamente a las matrices R y Q sin formar H_k, vectorizanco 
            # actualizao R
            uR = u @ R[k:, k:]
            u = u.reshape(-1, 1)   
            uR = uR.reshape(1, -1)
            R[k:, k:] -= 2.0 * (u @ uR)

            # actualizo Q
            Qu = Q[:, k:] @ u
            Qu = Qu.reshape(-1, 1)   
            u = u.reshape(1, -1)
            Q[:, k:] -= 2.0 * (Qu @ u)

    R_economica = R[:n, :] 
    Q_economica = Q[:, :n]
    
    return Q_economica, R_economica
    



# 1. 🏁 INICIO de la medición
tiempo_inicio = time.perf_counter()

# 2. ✅ FIN de la medición (Éxito)
tiempo_fin = time.perf_counter()
print(tiempo_fin - tiempo_inicio)