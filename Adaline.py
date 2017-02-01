'''
Created on 12/07/2016

@author: fabiana
'''
import numpy as np

# ------ Adaline implementado para disciplina de IA -----#
# ----- Alunos: Fabiana, Matheus -----#
class Adaline(object):
    
    def __init__(self,  InputSize, m):
        '''
        Constructor da Classe
        '''
        # ------- Define os parametros iniciais ------ #
        self.inputSize = InputSize
        self.theta = np.random.rand(self.inputSize + 1)     
        self.h0 = 0
        self.inputx = 0
        # ---- Funcao de ativacao sigmoide --- #
    def ativacao(self, z0):
        
        atv = np.where(z0 >= 1, 1, 0)
        return atv
        ''' 
            sig = 1/(1+np.exp(-z0))
            return sig
        '''    
    # ---- Progacao de valores ----- #
    def propagacaoValores(self, inputX):
        bias = np.ones((inputX.shape[0], 1))
        self.inputx = np.hstack((bias, inputX))
        #print "self.X\n", self.X
        z0 = np.dot(self.inputx, self.theta.T)
        #print "Z ", z0
        h0 = self.ativacao(z0)
        return z0, h0
    
    # --- Funcao de custo da rede -----#
    def funcaoCusto(self, X, y):
        z0,_ = self.propagacaoValores(X)
        J = 0.5* sum(sum( (z0-y) ** 2 ))
        return J
    
    #--- Realiza o Treinamento --- #
    def trainer(self, inputX, y, learningRate):
        z0,_ = self.propagacaoValores(inputX)
        erro = (z0 - y)
        #print'erro', erro
        delta = erro * self.inputx
        self.theta += -learningRate * delta[0]

                 

def main():
    # Entradas para o treinamento
    X = np.array(([0,0], [0,1], [1,0], [1,1]), dtype= float)
    y = np.array(([0],[0],[0],[1]), dtype = float)
    
    
    #tamanho da entrada
    inputSize = 2
    #numero de exemplos 
    m = len(X)
    #taxa de aprendizado
    learningRate = 0.4
    epochs = 200
    parada = 0.00001
    
    adaline = Adaline(inputSize, m)
    print '# ------- Valores Iniciais de Theta ------- # \n', adaline.theta
   
    print 'Realizando treinamento ....\n'
    
    for i in range(epochs):
        custoAn = adaline.funcaoCusto(X, y)
        print ' ---- Custo Anterior ---\n', custoAn
        for i in range(m):
            inputX = np.array(([X[i]]), dtype = float)
            adaline.trainer(inputX, y[i], learningRate)
        custoAtual = adaline.funcaoCusto(X,y)
        #print ' ---- Custo Atual ', custoAtual
        print '(custoAtual - custoAn) ',(custoAtual - custoAn) 
        
        if(abs((custoAtual - custoAn)) <= parada):
            print 'Rede Treinada!!\n'
            print ' ---- Custo Final ---\n', custoAtual
            break
        
    print '\n\n------ TREINAMENTO REALIZADO !! =] ----\n'
    print '# ---- Custo Final --- #\n', custoAtual
    print '# ------- Valores Finais de Theta ------- # \n', adaline.theta
    for i in range(m):
            inputX = np.array(([X[i]]), dtype = float)
            print 'X ', inputX 
            _,result =  adaline.propagacaoValores(inputX)
            print '\n ', result
if __name__ == '__main__':
    main()
        
        