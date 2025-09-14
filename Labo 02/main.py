import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def pointsCirc():
    centro = np.array([50,50])
    radio = 20       
    n_puntos = 100   

    # ángulos equiespaciados
    theta = np.linspace(0, 2*np.pi, n_puntos, endpoint=False)

    # coordenadas x e y
    x = centro[0] + radio * np.cos(theta)
    y = centro[1] + radio * np.sin(theta)

    # armar lista de puntos 2xN
    circ = np.vstack((x, y))
    return circ

def pointsSemiCirc():
    centro = np.array([0,0])
    radio = 20       
    n_puntos = 100   

    # ángulos equiespaciados para todo el círculo
    theta = np.linspace(0, 2*np.pi, n_puntos, endpoint=False)

    # coordenadas x e y
    x = centro[0] + radio * np.cos(theta)
    y = centro[1] + radio * np.sin(theta)

    # filtrar solo la mitad superior (y >= centro[1])
    mask = y >= centro[1]
    x = x[mask]
    y = y[mask]

    # armar lista de puntos 2xN
    semi_circ = np.vstack((x, y))
    return semi_circ

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = T@wz
    return xy

          
def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    plt.show()

    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    T = pd.read_csv('/home/manu/Escritorio/Talleres ALC/Labo 02/T.csv', header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    
    ## Ejercicio 1
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')

    T = np.array([[0.5,0],[0,0.5]])
    vistform(T, wz, 'Encoger coordenadas')
    
    ## Ejercicio 2
    a = 2; b = 3
    T = np.array([[a,0],[0,b]])
    T_prima = np.array([[1/a,0],[0,1/b]])

    vistform(T, np.array([[2], [14]]), 'Reescalamiento de un punto')
    
    wz = pointsCirc()
    vistform(T, wz, 'Reescalamiento de circulo')

    ## Ejercicio 3
    
    b = 0; c = 0; 
    T = np.array([[1,b],[c,1]])
    T_prima = np.array([[-1/(b*c-1),b/(b*c-1)],[c/(b*c-1),-1/(b*c-1)]])

    vistform(T, wz, 'Con b = 0')
    
    b = 1; c = 0; 
    T = np.array([[1,b],[c,1]])
    vistform(T, wz, 'Con b = 1')
    
    ## Ejercicio 4

    alfa = 0
    T = np.array([[np.cos(alfa),-np.sin(alfa)],[np.sin(alfa),np.cos(alfa)]])
    for i in range (5):
        alfa = 0 + np.pi*i/2
        T = np.array([[np.cos(alfa),-np.sin(alfa)],[np.sin(alfa),np.cos(alfa)]])
        vistform(T, pointsSemiCirc(), 'Con alfa = '+ str(0 + np.pi*i/2))

    ## Ejercicio 5

    alfa = np.pi/4
    T = (np.array([[np.cos(alfa),-np.sin(alfa)],
                   [np.sin(alfa),np.cos(alfa)]]) @ 
        np.array([[2,0],
                  [0,3]]) @
        np.array([[np.cos(alfa),np.sin(alfa)],
                  [-np.sin(alfa),np.cos(alfa)]]))
    vistform(T, pointsCirc(), 'Sobre circulo')

    
if __name__ == "__main__":
    main()
