from data_utils import load_dataset
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import dot
import pandas as pd # Only for formatting and plotting

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#e7ba52', '#dbdb8d']

class MaunaLoa:
    def __init__(self):
        '''
        Load dataset mauna_loa
        '''
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset('mauna_loa')
        self.dimension = np.shape(self.x_test)[1]
        self.num_trainSet = np.shape(self.x_train)[0]
        self.num_validSet = np.shape(self.x_valid)[0]
        self.num_testSet = np.shape(self.x_test)[0]

    def setParameters(self, M, lamb, model):
        '''
        Set the parameters that requires tuning here as class attributes
        '''
        self.M = M
        self.lamb = lamb
        self.model = model


    def normalization(self, x_set):
        '''
        Returned a matrix of the input dataset normalized against its mean and stddev
        '''
        mean = x_set.mean(axis=0, keepdims=True)
        stddev = x_set.std(axis=0, keepdims=True)
        return (x_set - mean)/stddev


    def getPhi(self, x):
        '''
        Returns a row vector [phi_0(x) phi_1(x) ... phi_{M-1}(x)]
        INPUT x: a column vector of length = self.dimension. In this case of mauna_loa, just a scalar
        INPUT M: dimension of the output row vector
        INPUT: model: basis function models, such as 'polynomial', 'gaussian', 'fourier'
        '''
        row = [1] # phi_0 is always 1
        if self.model == 'gaussian': s = (max(self.x_train)-min(self.x_train))/(self.M-2)
        for i in range(1, self.M):
            if self.model == 'polynomial':
                phi_i = pow(x, i)
            if self.model == 'gaussian':
                mu_i = min(self.x_train) + (i-1)*s
                phi_i = math.exp(-pow((x-mu_i), 2)/(2*pow(s, 2)))
            row.append(phi_i)
        return row


    def getPhiMatrix(self, set):
        return np.array([self.getPhi(x) for x in set])


    def getWeight(self):
        '''
        Uses the economic SVD method to compute vector w for f(X, w) = phiMatrix * w
        MUST call getWeight before any testing, since self.w is defined in this function
        OUTPUT: w = V()
        '''
        self.phiMatrix = self.getPhiMatrix(self.x_train)
        U, s, VT = np.linalg.svd(self.phiMatrix, full_matrices=False) # The economy SVD (if want full SVD, change the second parameter to True)
        S = np.zeros((U.shape[1], VT.shape[0]))
        S[:VT.shape[0], :VT.shape[0]] = np.diag(s)
        ST, V, UT = S.T, VT.T, U.T
        lambI = np.identity(U.shape[1]) * self.lamb

        w = V.dot(np.linalg.inv(ST.dot(S)+lambI)).dot(ST).dot(UT).dot(self.y_train)
        self.w = w


    def plotRegression(self, set, M=1, lamb =1, model='polynomial'):

        if set == 'validation ':
            x, y_actual = self.x_valid, self.y_valid
        elif set == 'test':
            x, y_actual = self.x_test, self.y_test
        elif set == 'train':
            x, y_actual = self.x_train, self.y_train
        self.getWeight()
        y_predicted = self.getPhiMatrix(x).dot(self.w)
        rmse = np.sqrt(pow(np.array(y_predicted-y_actual), 2).mean())
        # print('RMSE is', rmse, 'for data', self.dataset, 'with linear regression on the', set, 'set.')

        plt.style.use('bmh')
        plt.scatter(x[:, 0], y_actual[:, 0])
        plt.plot(x[:, 0], y_predicted[:, 0], color='red')
        plt.title('GLM on the %s set of "mauna_loa" \n Resulting rmse = %1.4f' %(set, rmse), loc='center', size=12)
        plt.show()



if __name__ == '__main__':
    Q1 = MaunaLoa()
    Q1.setParameters(M=100, lamb=0, model='gaussian')
    Q1.plotRegression('validation')
    print(Q1.x_train)
    print('phiMatrix:', Q1.phiMatrix)
    print('w:', Q1.w)
