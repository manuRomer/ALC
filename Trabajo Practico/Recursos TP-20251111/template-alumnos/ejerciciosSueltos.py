from alc import *
import numpy as np
import os

carpetaGatosYPerros = '/home/manu/Escritorio/Talleres ALC/Trabajo Practico/Recursos TP-20251111/template-alumnos/template-alumnos/dataset/cats_and_dogs'
# Ejercicio 1
#aca meti un poco de ChatGPT porque no sabia muy bien como cargar los datos
#como no podemos testearlo, no se que onda esta implementacion
def cargarDataset(carpeta):
    #root = "template-alumnos/template-alumnos/dataset/cats_and_dogs"
    #Xt, Yt, Xv, Yv = cargarDataset(root)

    # Armo la ruta completa a la carpeta de cats_and_dogs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    carpetaGatosYPerros = os.path.join(base_dir, carpeta)
    # carpetas
    train_cats = os.path.join(carpetaGatosYPerros, "train", "cats", "efficientnet_b3_embeddings.npy")
    train_dogs = os.path.join(carpetaGatosYPerros, "train", "dogs", "efficientnet_b3_embeddings.npy")
    val_cats   = os.path.join(carpetaGatosYPerros, "val", "cats", "efficientnet_b3_embeddings.npy")
    val_dogs   = os.path.join(carpetaGatosYPerros, "val", "dogs", "efficientnet_b3_embeddings.npy")

    Xtc = np.load(train_cats)   # gatos train
    Xtd = np.load(train_dogs)   # perros train
    Xvc = np.load(val_cats)     # gatos val
    Xvd = np.load(val_dogs)     # perros val

    Xt = traspuesta(np.concatenate([Xtc, Xtd], axis=0))
    Xv = traspuesta(np.concatenate([Xvc, Xvd], axis=0))

    Nc_train = Xtc.shape[0]   
    Nd_train = Xtd.shape[0]  

    Yt_cats = np.zeros((2, Nc_train))
    Yt_dogs = np.zeros((2, Nd_train))

    for i in range(Nc_train):
        Yt_cats[0, i] = 1
        Yt_cats[1, i] = 0

    for i in range(Nd_train):
        Yt_dogs[0, i] = 0
        Yt_dogs[1, i] = 1

    Yt = np.concatenate([Yt_cats, Yt_dogs], axis=1)

    Nc_val = Xvc.shape[0]  
    Nd_val = Xvd.shape[0]  
    
    Yv_cats = np.zeros((2, Nc_val))
    Yv_dogs = np.zeros((2, Nd_val))

    for i in range(Nc_val):
        Yv_cats[0, i] = 1
        Yv_cats[1, i] = 0

    for i in range(Nd_val):
        Yv_dogs[0, i] = 0
        Yv_dogs[1, i] = 1

    Yv = np.concatenate([Yv_cats, Yv_dogs], axis=1)

    return Xt, Yt, Xv, Yv

# Ejercicio 2 
def pinvEcuacionesNormales(X,L,Y):
    n, p = X.shape
    X_t = traspuesta(X)
    L_t = traspuesta(L)
    if n > p:
        # Asumo que L = cholesky(X^T @ X)
        # Quiero resolver L @ L^T @ U = X^T 
        Z = np.zeros((p, n))
        U = np.zeros((p, n))
        
        # Paso intermedio. Sustitucion hacia adelante: L @ Z = X^T
        for i in range(n):
            Z[:, i] = res_tri(L, X_t[:, i], True)
            
        # Resuelvo el sistema. Sustitución hacia atrás: L^T @ U = Z
        for i in range(n):
            U[:, i] = res_tri(L_t, Z[:, i], False)

        # Calculo W
        W = multi_matricial(Y, U)
    
    elif n < p:
        # Asumo que L = cholesky(X @ X^T)
        # Quiero resolver V @ X @ X^T = X^T
        # Para usar res_tri tengo que resolver a derecha asi que aplico traspuesta a ambos lados y resuelvo L @ L^T @ V^T = X
        
        Z = np.zeros((n, p))
        Vt = np.zeros((n, p))

        # Paso intermedio. Sustitucion hacia adelante: L @ Z = X
        for i in range(p):
            Z[:, i] = res_tri(L, X[:, i], True)
            
        # Resuelvo el sistema. Sustitución hacia atrás: L^T @ V^T = Z
        for i in range(p):
            Vt[:, i] = res_tri(L_t, Z[:, i], False)
        
        V = traspuesta(Vt)
        
        W = multi_matricial(Y, V)
        
    else:
        # Como la pseudoinversa X^+ = X^-1 entonces W = Y @ X^-1
        W = multi_matricial(Y, inversa(X))
            
    return W

# Ejercicio 3
def pinvSVD(U, S, V, Y):
    n = S.shape[0]

    # Calculamos Sigma_1^-1
    S_1 = inversaDeMatrizDiagonal(S[:, :n])

    # Calculamos la pseudo-inversa de X
    V_1 = V[:,:n]
    U_1 = U[:,:n]
    pseudoInversa = multi_matricial(multi_matricial(V_1, S_1), traspuesta(U_1))
    
    W = multi_matricial(Y, pseudoInversa)

    return W

# Ejercicio 4
def pinvHouseHolder(Q,R,Y):
    return calcularWconQR(Q, R, Y)

def pinvGramSchmidt(Q,R,Y):
    return calcularWconQR(Q, R, Y)

# Ejercicio 5
def esPseudoInversa(X, pX, tol = 1e-8):
    #La pseudo inversa es la unica matriz que cumple los 4 puntos mencionados en el tp (al final de la pagina 3)
    # 1) X @ pX @ X = X
    if not matricesIguales(multi_matricial(X, multi_matricial(pX, X)), X, tol):
        return False
    # 2) pX X pX = pX
    if not matricesIguales(multi_matricial(pX, multi_matricial(X, pX)), pX, tol):
        return False
    # 3) (X pX)^T = X pX
    XpX = multi_matricial(X, pX)
    if not matricesIguales(traspuesta(XpX), XpX, tol):
        return False
    # 4) (pX X)^T = pX X
    pXX = multi_matricial(pX, X)
    if not matricesIguales(traspuesta(pXX), pXX, tol):
        return False
    return True

# Ejercicio 6 
def evaluacion():
    Xt, Yt, Xv, Yv = cargarDataset(carpetaGatosYPerros)

    # En el contexto del TP, n < p entonces para el algoritmo 1 aplicamos Cholesky sobre X @ X^T
    L = cholesky(multi_matricial(Xt, traspuesta(Xt)))
    WEN = pinvEcuacionesNormales(Xt, L , Yt)
    print('Termino WEN')
    
    U, S, V = svd_reducida(Xt)
    WSVD = pinvSVD(U, S, V, Yt)
    print('Termino WSVD')

    QHH, RHH = QR_con_HH(Xt)
    WQRHH =(QHH, RHH, Yt)
    print('Termino WQRHH')
    
    QGS, RGS = QR_con_GS(Xt)
    WQRGS =(QGS, RGS, Yt)
    print('Termino WQRGS')
    
    

evaluacion()