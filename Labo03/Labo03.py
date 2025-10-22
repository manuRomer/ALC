import numpy as np

def pointsCirc():
    centro = np.array([0, 0])
    radio = 5
    n_puntos = 360

    # ángulos equiespaciados
    theta = np.linspace(0, 2*np.pi, n_puntos, endpoint=False)

    # coordenadas x e y
    x = centro[0] + radio * np.cos(theta)
    y = centro[1] + radio * np.sin(theta)

    # armar lista de puntos 2xN
    circ = [np.array([xi, yi]) for xi, yi in zip(x, y)]
    return circ

circulo = pointsCirc()

## Ejercicio 1

def norma(x, p):
    if p == 'inf':
        return np.max(np.abs(x))
    else:
        return (np.sum(np.abs(x)**p))**(1/p)

def normaliza(X, p):
    Y = []
    for vector in X:
        n = norma(vector, p)
        Y.append(vector / n)
    return Y

# arrays = [normaliza(circulo, 1), normaliza(circulo, 2), normaliza(circulo, 5),
#           normaliza(circulo, 10), normaliza(circulo, 100), normaliza(circulo, 'inf')]
# colores = ['r', 'g', 'b', 'c', 'm', 'y']  # cada norma con un color distinto
# labels = ['p=1', 'p=2', 'p=5', 'p=10', 'p=100', 'p=200']

# plt.figure(figsize=(6,6))
# for arr, c, lab in zip(arrays, colores, labels):
#     arr = np.array(arr).T  # ahora es 2xN
#     plt.plot(arr[0,:], arr[1,:], '.', color=c, label=lab)

# plt.gca().set_aspect('equal')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Vectores normalizados según diferentes normas p')
# plt.show()

## Ejercicio 2

def normaMatMC(A,q,p,Np):
    norma_max = 0
    for _ in range(Np):
        n = A.shape[1]
        x = 2*np.random.rand(n) - 1
        x_normalizado = x / norma(x, q)
        valor =  norma(A@x_normalizado, p)
        if valor > norma_max:
            norma_max = valor
    return norma_max

def normaExacta(A,p):
    if p == 1:
        sumas_columnas = np.sum(np.abs(A), axis=0) 
        return np.max(sumas_columnas)
    if p == 'inf':
        sumas_filas = np.sum(np.abs(A), axis=1) 
        return np.max(sumas_filas)
    

## Ejercicio 3

def condMC(A,p):
    inversa = np.linalg.inv(A)
    return normaMatMC(A,p,p,100) * normaMatMC(inversa,p,p,100)

def variaPerc(b, perc):
    factores = 1 + (np.random.rand(len(b)) * 2 - 1) * (perc / 100)
    b_var = b * factores
    return b_var

def dos_b_ii(A, p, perc, Np):
    n = A.shape[0]

    Bs = [np.random.rand(n) for _ in range(Np)]       # lista con Np vectores b
    Bs_prima = [variaPerc(b, perc) for b in Bs]       # lista con los Np vectores b'

    res_Bs = [np.linalg.solve(A, b) for b in Bs]
    res_Bs_prima = [np.linalg.solve(A, b_prima) for b_prima in Bs_prima]

    Ex = [norma(res_Bs[i]-res_Bs_prima[i], p)/norma(res_Bs[i], p) for i in range(Np)]
    Eb = [norma(Bs[i]-Bs_prima[i], p)/norma(Bs[i], p) for i in range(Np)]
    
    indice_max = np.argmax(Ex)

    return Ex, Eb, Bs[indice_max], Bs_prima[indice_max]
