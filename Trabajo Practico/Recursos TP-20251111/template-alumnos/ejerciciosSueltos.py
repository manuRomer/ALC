from modulo_ALC import *
import numpy as np
import math
from Labo08 import svd_reducida
import os


# Ejercicio 1
#aca meti un poco de ChatGPT porque no sabia muy bien como cargar los datos
#como no podemos testearlo, no se que onda esta implementacion
def cargarDataset(carpeta):
    #root = "template-alumnos/template-alumnos/dataset/cats_and_dogs"
    #Xt, Yt, Xv, Yv = cargarDataset(root)
    
    # carpetas
    train_cats = os.path.join(carpeta, "train", "cats", "efficientnet_b3_embeddings.npy")
    train_dogs = os.path.join(carpeta, "train", "dogs", "efficientnet_b3_embeddings.npy")
    val_cats   = os.path.join(carpeta, "val", "cats", "efficientnet_b3_embeddings.npy")
    val_dogs   = os.path.join(carpeta, "val", "dogs", "efficientnet_b3_embeddings.npy")

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

# Ejercicio 2 (falta acomodar)
def pinvEcuacionesNormales(X,L,Y):
    n = X.shape[0]
    p = X.shape[1]
    if n > p:
        X_t = traspuesta(X)
        X_t_X = multi_matricial(X_t, X)
        L_t = traspuesta(L)
        Z = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            Z[i] = sustitucionParaAdelante(L, X_t[i])
            
        U = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            U[i] = sustitucionParaAtras(L_t, Z[i])
        W = multi_matricial(Y, U)
    
    elif n < p:
        X_t = traspuesta(X)
        X_t_X = multi_matricial(X_t, X)
        L_t = traspuesta(L)
        Z = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            Z[i] = sustitucionParaAdelante(L, X_t[i])
            
        Vt = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            Vt[i] = sustitucionParaAtras(L_t, Z[i])
        
        Vt = traspuesta(Vt)
        
        W = multi_matricial(Y, Vt)
        
    else:
        return multi_matricial(Y, inversa(X))
            
    return W

# Ejercicio 3
def pinvSVD(U, S, V, Y):
    N = S.shape[0]
    
    V1 = V[:,:N]
    U1 = U[:,:N]
    pseudoInversa = multi_matricial(multi_matricial(V1, S), traspuesta(U1))
    
    W = multi_matricial(Y, pseudoInversa)
    return W

# Ejercicio 4
def pinvHouseHolder(Q,R,Y):
    Vt = np.zeros((R.shape[1], R.shape[1]))
    for i in range(R.shape[1]):
        Vt[i] = sustitucionParaAdelante(R, Q[i])

    return multi_matricial(Y, traspuesta(Vt))

def pinvGramSchmidt(Q,R,Y):
    Vt = np.zeros((R.shape[1], R.shape[1]))
    for i in range(R.shape[1]):
        Vt[i] = sustitucionParaAdelante(R, Q[i])

    return multi_matricial(Y, traspuesta(Vt))

# Ejercicio 5
def esPseudoInversa(X, pX, tol = 1e-8):
    #La pseudo inversa es la unica matriz que cumple los 4 puntos mencionados en el tp (al final de la pagina 3)
    # 1) X pX X = X
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

# Ejercicio 6 (falta hacer la matriz de confusion, es la matriz de 2x2 que te dice que tanto le erraste)
# para cada W, hay que hacer WXv y ver si dio el resultado esperado.
def evaluacion():
    Xt, Yt, Xv, Yv = cargarDataset()#template-alumnos/template-alumnos/dataset/cats_and_dogs)
    LCholesky = cholesky(Xt)
    WCholesky = pinvEcuacionesNormales(Xt, LCholesky, Yt)
    
    U, Epsilon, V = svd_reducida(Xt)
    WSVD = pinvSVD(U, Epsilon, V, Yt)

    Qhh, Rhh = QR_con_HH(Xt)
    WQRHH =(Qhh, Rhh, Yt)
    
    Qgs, Rgs = QR_con_GS(Xt)
    WQRGS = pinvGramSchmidt(Qgs, Rgs, Yt) 



    