from data_utils import load_dataset
import matplotlib.pyplot as plt
import math
import numpy as np
from sympy.matrices import GramSchmidt
from numpy import dot
import pandas as pd # Only for formatting and plotting

from Kernels import *
from BasisFunctions import *

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#e7ba52', '#dbdb8d']

class GLM:
    def __init__(self, datasetName):
        '''
        Load dataset mauna_loa
        '''
        self.dataset = datasetName
        if datasetName == 'rosenbrock':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test =  load_dataset(datasetName, n_train=1000, d=2)
        else:
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(datasetName)
        self.dimension = np.shape(self.x_test)[1]
        self.num_trainSet = np.shape(self.x_train)[0]
        self.num_validSet = np.shape(self.x_valid)[0]
        self.num_testSet = np.shape(self.x_test)[0]


    def initCrossValidation(self):
        '''
        Combine the training and validation set together for the splitting in cross-validation
        '''
        self.x_all = np.concatenate([self.x_train, self.x_valid])
        self.y_all = np.concatenate([self.y_train, self.y_valid])
        self.index_all = list(range(np.shape(self.x_all)[0]))
        np.random.seed(99)
        np.random.shuffle(self.index_all)

        self.num_validSet = round(np.shape(self.x_all)[0]/5)
        self.num_trainSet = np.shape(self.x_all)[0] - self.num_validSet


    def splitCrossValidation(self, foldIndex):
        '''
        Split data into two sets of ratio 4:1 according to the foldIndex
        INPUT: allData: concatenate dataset
        INPUT: foldIndex: from 1 to 5, decides how to partition the dataset (must be from outside the class)
        OUTPUT: train, set: the 4:1 ratio datasets
        '''
        total = len(self.index_all)
        oneFifth = round(total/5)
        if foldIndex in [1, 2, 3, 4]:
            index_train = self.index_all[:oneFifth*(foldIndex-1)] + self.index_all[oneFifth*foldIndex:]
            index_valid = self.index_all[oneFifth*(foldIndex-1) : oneFifth*foldIndex]
        elif foldIndex == 5: # for the last fold, cound backwards so that it has the same number of data as the other folds
            index_train = self.index_all[:(total-oneFifth)]
            index_valid = self.index_all[(total-oneFifth):]
        self.x_train, self.x_valid = self.x_all[index_train], self.x_all[index_valid]
        self.y_train, self.y_valid = self.y_all[index_train], self.y_all[index_valid]


    def setParameters(self, method='basisfunc', model='polynomial', lamb=0, M=100, theta=0.1, degree=2):
        '''
        Set the parameters that requires tuning here as class attributes
        '''
        self.method = method
        self.model = model
        self.lamb = lamb
        self.M = M
        self.theta = theta
        self.degree = degree


    def normalization(self, x_set):
        '''
        Returned a matrix of the input dataset normalized against its mean and stddev
        '''
        mean = x_set.mean(axis=0, keepdims=True)
        stddev = x_set.std(axis=0, keepdims=True)
        return (x_set - mean)/stddev


    def getPrediction(self, x_set, y_set):
        '''
        Pass in the datasets and output the prediction set and rmse
        '''
        if self.method == 'basisfunc':
            method = BasisFunctions(self.model, self.lamb, self.M, self.degree)
            method.getWeight(self.x_train, self.y_train)
            self.w, self.phiMatrix = method.w, method.phiMatrix
            y_predicted = method.getPhiMatrix(x_set).dot(method.w)

        elif self.method == 'kernel':
            method = Kernels(self.M, self.model, self.lamb, self.theta, self.degree)
            method.getAlpha(self.x_train, self.y_train)
            self.w, self.K = method.w, method.K
            y_predicted = method.getGram(self.x_train, x_set).T.dot(method.alpha)

        rmse = np.sqrt(pow(np.array(y_predicted-y_set), 2).mean())
        return y_predicted, rmse


    def runRegression(self, set):
        '''
        Show and return the prediction results
        INPUT: set: can be one of 'cross-validation', 'validation', 'test', or 'train'
        '''
        if set == 'cross-validation':
            X, Y_actual, Y_predicted, RMSE = np.array([[None]]), np.array([[None]]), np.array([[None]]), 0
            self.initCrossValidation()
            for foldIndex in range(1, 6):
                self.splitCrossValidation(foldIndex)
                x, y_actual = self.x_valid, self.y_valid
                X = np.concatenate((X, x))
                Y_actual = np.concatenate((Y_actual, y_actual))
                y_predicted, rmse = self.getPrediction(x, y_actual)
                Y_predicted = np.concatenate((Y_predicted, y_predicted))
                RMSE += rmse/5.
        elif set == 'validation':
            X, Y_actual = self.x_valid, self.y_valid
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
        elif set == 'test':
            X, Y_actual = self.x_test, self.y_test
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
        elif set == 'train':
            X, Y_actual = self.x_train, self.y_train
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)

        plt.style.use('bmh')
        plt.scatter(X[:, 0], Y_actual[:, 0], s=2, color=_COLORS[2])
        plt.scatter(X[:, 0], Y_predicted[:, 0], s=2, color=_COLORS[0])
        plt.legend(('Training', 'Prediction'))
        # plt.plot(X[:, 0], Y_predicted[:, 0], linewidth=1, color=_COLORS[0])
        plt.title('GLM on the %s set of "%s" with %s\n Resulting rmse = %1.4f' %(set, self.dataset, self.method, RMSE), loc='center', size=12)
        plt.show()
        return RMSE



def Q1():
    Q1 = GLM('mauna_loa')
    # Q1.setParameters(method='basisfunc', model='polynomial', lamb=0, degree=5)
    Q1.setParameters(method='basisfunc', model='gaussian', lamb=0, M=100)
    # Q1.runRegression('cross-validation')
    Q1.runRegression('train')

    print(Q1.x_train)
    print('phiMatrix:', Q1.phiMatrix)
    print('w:', Q1.w)

def Q2():
    Q2 = GLM('mauna_loa')
    print(Q2.x_train.shape, Q2.x_test.shape)
    Q2.setParameters(method='kernel', model='gaussian', lamb=0.0001, M=100, theta=0.1, degree=2)
    Q2.runRegression('cross-validation')


if __name__ == '__main__':
    Q1()
