
def reflexionHH(v):
    """Devuelve la matriz de Householder que refleja v sobre e1"""
    n = v.shape[0]
    e1 = np.zeros(n)
    e1[0] = 1.0

    # vector de Householder estable
    sign = 1.0 if v[0] >= 0 else -1.0


    u = v + sign * norma(v, 2) * e1
    u = u / norma(u, 2)
    H = np.eye(n) - 2 * multi_matricial(traspuesta(u), u)
    return H

def diagRH(A,tol=1e-15,K=1000):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if n == 1:
        return np.eye(1), A.copy()

    # 1️⃣ Método de la potencia: autovector y autovalor dominante
    v, lambda1, _ = metpot2k(A, tol)
    v = v / norma(v, 2)

    # 2️⃣ Householder que lleva v → e1
    H = reflexionHH(v)

    # 3️⃣ Aplicar transformación
    B = multi_matricial(H, multi_matricial(A, traspuesta(H))) 

    # 4️⃣ Diagonalizar recursivamente la submatriz inferior derecha
    A_sub = B[1:, 1:]
    S_sub, D_sub = diagRH(A_sub, tol, K)

    # 5️⃣ Armar S y D globales
    D = np.zeros((n, n))
    D[0, 0] = lambda1
    D[1:, 1:] = D_sub

    S = np.eye(n)
    S[1:, 1:] = S_sub
    S = multi_matricial(H, S)

    return S, D