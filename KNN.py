'''
Created on 27/05/2016

@author: Fabiana Zioti | Matheus Reis

Algoritmo KNN E 1NN
sendo que para 1NN basta mudar o valor de k para 1
'''


import numpy as np
import math
import matplotlib.pyplot as plt

def distanciaEuclidiana(novo, trainX):
    distancia = np.sum (((novo - trainX) ** 2), axis = 1)
    print 'Distancia Euc', distancia
    return np.sqrt(distancia)

def votos(y, menores):
    cont0 = 0
    cont1 = 0
    
    for x in menores:
        print 'y', y[x]
        if y[x] == 0:
            cont0 +=1
        else:
            cont1 +=1
    
    if cont0 > cont1:
        return 0
    else:
        return 1
    
def main():
    
    # gera os valores de x e y de modo randomico
    trainX = np.random.randint(0,100,(25,2)).astype(np.float)
    print trainX
    y = np.random.randint(0,2,(25,1)).astype(np.float)
    
    # define o numero de vizinhos 
    k = 1
    
    '''
        vermelho eh 0
        azul eh 1
    '''
    # Pega os vermelhos e plota eles
    red = trainX[y.ravel() == 0]
    plt.scatter(red[:,0],red[:,1],80,'r','^')
    
    # Pega os azuis e plota eles
    blue = trainX[y.ravel()==1]
    plt.scatter(blue[:,0],blue[:,1],80,'b','s')
    
    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
    
    #plt.show()
    

    distanciaEu = distanciaEuclidiana(newcomer, trainX)
    
    menores = []
    for i in xrange(k):
        posi = distanciaEu.argmin()
	print posi
        menores.append(posi)
        distanciaEu[posi] = 10000000000
    
    print 'Posicao Menores Distancias',menores
    
    result = votos(y, menores)
    
    print 'Result', result
    
    plt.title('Algoritmo KNN Plot da Base')
    plt.show()
    


if __name__ == '__main__':
    main()
    
