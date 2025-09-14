import numpy as np

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


def matricesIguales(A, B):
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
    return np.diag(s)


def rota_y_escala(theta, s):
    """
    Recibe un ángulo theta y una tira de números s,
    y retorna una matriz de 2x2 que rota el vector en un ángulo theta
    y luego lo escala en un factor s.
    """
    return rota(theta) @ escala(s)


def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo mueve en un valor fijo b.
    """
    A = rota_y_escala(theta, s)
     # Construyo matriz 3x3 identidad
    T = np.eye(3)
    # Inserto A en la parte superior izquierda
    T[:2,:2] = A
    # Inserto b en la última columna
    T[:2,2] = b
    return T


def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un ángulo theta,
    una tira de números s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformación afín a v.
    """
    vh = np.array([v[0], v[1], 1])      # vector homogéneo
    wh = afin(theta, s, b) @ vh         # aplicar transformación
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
    Y = []
    for vector in X:
        n = norma(vector, p)
        Y.append(vector / n)
    return Y

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q, p} y el vector x en el cual se alcanza
    el máximo.
    """
    norma_max = 0; x_max = None
    for _ in range(Np):
        n = A.shape[1]
        x = 2*np.random.rand(n) - 1
        x_normalizado = x / norma(x, q)
        valor =  norma(A@x_normalizado, p)
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

def condMC(A, p):
    """
    Devuelve el número de condición de A usando la norma inducida p.
    """
    inversa = np.linalg.inv(A)
    return normaMatMC(A,p,p,100) * normaMatMC(inversa,p,p,100)

def condExacta(A, p):
    """
    Devuelve el número de condición de A a partir de la fórmula de
    la ecuación (1) usando la norma p.
    """
    inversa = np.linalg.inv(A)
    return normaExacta(A,p) * normaExacta(inversa,p)
    return  