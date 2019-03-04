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


    def initTest(self):
        '''
        Combine the training and validation set together as a whole training set for testing
        '''
        self.x_train = np.concatenate([self.x_train, self.x_valid])
        self.y_train = np.concatenate([self.y_train, self.y_valid])
        self.num_trainSet = np.shape(self.x_train)[0]


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
        print(lamb)
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
            method = BasisFunctions(self.x_train, self.model, self.lamb, self.M, self.degree)
            method.getWeight(self.x_train, self.y_train)
            self.w, self.phiMatrix = method.w, method.phiMatrix
            y_predicted = method.getPhiMatrix(x_set).dot(method.w)

        elif self.method == 'kernel':
            method = Kernels(self.M, self.model, self.lamb, self.theta, self.degree)
            method.getAlpha(self.x_train, self.y_train)
            self.alpha, self.K = method.alpha, method.K
            y_predicted = method.getGram(self.x_train, x_set).T.dot(method.alpha)

        rmse = np.sqrt(pow(np.array(y_predicted-y_set), 2).mean())
        return y_predicted, rmse


    def runRegression(self, set, graph='on'):
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
            markersize = 2
        elif set == 'validation':
            X, Y_actual = self.x_valid, self.y_valid
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 20
        elif set == 'test':
            self.initTest()
            X, Y_actual = self.x_test, self.y_test
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 20
        elif set == 'train':
            X, Y_actual = self.x_train, self.y_train
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 2

        if graph == 'on':
            plt.style.use('bmh')
            plt.scatter(X[:, 0], Y_actual[:, 0], s=markersize, color=_COLORS[2])
            plt.scatter(X[:, 0], Y_predicted[:, 0], marker = '*', s=markersize, color=_COLORS[0])
            plt.legend(('Actual', 'Prediction'))
            # plt.plot(X[:, 0], Y_predicted[:, 0], linewidth=1, color=_COLORS[0])
            plt.title('GLM on the %s set of "%s" with %s\n Resulting rmse = %1.4f' %(set, self.dataset, self.method, RMSE), loc='center', size=12)
            plt.show()
        return RMSE


    def runClassification(self, set):
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
            markersize = 2
        elif set == 'validation':
            X, Y_actual = self.x_valid, self.y_valid
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 20
        elif set == 'test':
            X, Y_actual = self.x_test, self.y_test
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 20
        elif set == 'train':
            X, Y_actual = self.x_train, self.y_train
            Y_predicted, RMSE = self.getPrediction(X, Y_actual)
            markersize = 2

        class_predicted = []
        num_classes = np.shape(self.y_test)[1]
        for y in Y_predicted:
            maxClass = list(y).index(max(y))
            class_base = np.zeros(num_classes)
            class_base[maxClass] = 1
            class_predicted.append(class_base)

        # plt.style.use('bmh')
        # plt.scatter(X[:, 0], Y_actual[:, 0], s=markersize, color=_COLORS[2])
        # plt.scatter(X[:, 0], Y_predicted[:, 0], marker = '*', s=markersize, color=_COLORS[0])
        # plt.legend(('Actual', 'Prediction'))
        # plt.show()

        accuracies = [sum(list(class_predicted)[i]!=list(Y_actual)[i])/2 for i in range(np.shape(class_predicted)[0])]
        accuracy = 1-sum(accuracies)/len(accuracies)
        print('Accuracy is', str(accuracy*100)+'% for data', self.dataset, 'on the', set, 'set.')
        return accuracy



def Q1():
    glm = GLM('mauna_loa')
    # _LAMB = [0, 0.0001, 0.001, 0.05, 0.01, 0.05, 0.1, 1]
    # LAMB, RMSE, RMSEval = [], [], []
    # for lamb in _LAMB:
    #     LAMB.append(0)
    #     glm.setParameters(method='basisfunc', model='DIY', lamb=0, degree=4)
    #     RMSEval.append(glm.runRegression('validation', graph = 'off'))
    #     RMSE.append(glm.runRegression('test', graph = 'off'))
    # print(pd.DataFrame({'lambda': LAMB, 'validation': RMSEval, 'testing': RMSE}))
    # print('RMSE is the smallest for validation sets at lambda =', _LAMB[RMSEval.index(min(RMSEval))])
    # print('RMSE is the smallest for test sets at lambda =', _LAMB[RMSE.index(min(RMSE))])
    #
    # glm.setParameters(method='basisfunc', model='DIY', lamb=_LAMB[RMSE.index(min(RMSE))], degree=4)
    # glm.runRegression('test', graph = 'on')
    # for i in range(7):
    #     if i == 6:
    #         lamb = 0
    #     else: lamb=10**(-i)
    #     print('lamb =', lamb)
    #     glm.setParameters(method='basisfunc', model='DIY', lamb=lamb, degree=4)
    #     print(glm.runRegression('test', graph='on'))
    glm.setParameters(method='basisfunc', model='DIY', lamb=0.1, degree=4)
    print(glm.runRegression('test', graph='off'))

def Q2():
    Q2 = GLM('mauna_loa')
    print(Q2.x_train.shape, Q2.x_test.shape)
    Q2.setParameters(method='kernel', model='DIY', lamb=2, degree=4)
    Q2.runRegression('train', graph='off')

def Q3():
    # _THETA = [0.05, 0.1, 0.5, 1, 2]
    # _LAMB = [0.001, 0.01, 0.1, 1]
    # _DATASET = [('mauna_loa', 'regression'), ('rosenbrock', 'regression'), ('iris', 'classification')]
    _THETA = [0.05]
    _LAMB = [0.001]
    _DATASET = [('mauna_loa', 'regression'), ('rosenbrock', 'regression'), ('iris', 'classification')]
    RMSE = {}
    for (dataset, task) in _DATASET:
        print('Processing dataset', dataset, '...')
        RMSE[dataset] = {}
        glm = GLM(dataset)
        for theta in _THETA:
            RMSE[dataset][theta] = {}
            for lamb in _LAMB:
                glm.setParameters(method='kernel', model='gaussian', lamb=lamb, theta=theta)
                if task == 'regression':
                    RMSE[dataset][theta][lamb] = glm.runRegression('test')
                elif task == 'classification':
                    RMSE[dataset][theta][lamb] = glm.runClassification('test')
    RMSE_mauna_loa, RMSE_rosenbrock, RMSE_iris = pd.DataFrame(RMSE['mauna_loa']), pd.DataFrame(RMSE['rosenbrock']), pd.DataFrame(RMSE['iris'])
    return RMSE_mauna_loa, RMSE_rosenbrock, RMSE_iris


if __name__ == '__main__':
    Q1()
