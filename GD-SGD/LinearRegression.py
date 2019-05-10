from data_utils import load_dataset
import numpy as np
import scipy as sp
import pandas as pd
from numpy import dot
import matplotlib.pyplot as plt
import copy

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#e7ba52', '#e69fa5', '#dbdb8d']
_ITERCAP = 1000
_DATASIZE = 1000

class LinearRegression:
    def __init__(self):
        self.dataset = 'pumadyn32nm'
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test  = load_dataset(self.dataset)

        # Extract the first 1000 data points
        self.x_train, self.y_train = self.x_train[:_DATASIZE], self.y_train[:_DATASIZE]

        # Add x0 = 1 to each x vector
        x0 = np.ones((np.shape(self.x_train)[0], 1))
        self.x_train = np.concatenate((x0, self.x_train), axis=1)
        x0 = np.ones((np.shape(self.x_test)[0], 1))
        self.x_test = np.concatenate((x0, self.x_test), axis=1)

        self.num_train = _DATASIZE
        (self.num_test, self.num_dimension) = self.x_test.shape


    def preprocessSGD(self):
        indices = list(range(self.num_train))
        np.random.seed(666)
        np.random.shuffle(indices)
        self.x_train_Random, self.y_train_Random = self.x_train[indices], self.y_train[indices]


    def setParameters(self, convergenceError, learningRate, batchSize=1, GDvisualization=False):
        self.errorBound = convergenceError
        self.learningRate = learningRate
        self.stepsize = learningRate/self.num_train
        self.batchSize = batchSize
        self.num_batches = self.num_train/batchSize
        self.plotGD = GDvisualization


    def getLoss(self, w, X, Y):
        '''
        Function of column vector w, loss function for least square error
        Output: scaler value of loss
        '''
        return ((Y - X.dot(w)).T.dot(Y - X.dot(w))[0][0])/(2 * self.num_train)


    def getGradient(self, w, X, Y):
        '''
        Derivative/gradient of column vector w, loss function for least square error
        Output: column vector, the gradient
        '''
        return X.T.dot(X.dot(w) - Y)


    def getGDWeight(self):
        '''
        Get the exact gradient descent weight, full-batch
        Lerning rate is set to be a constant here
        '''
        w = np.zeros((self.num_dimension, 1))
        error, stablized, iter, ROUND, loss, LOSS = float('inf'), False, 0, [], float('inf'), []

        while error > self.errorBound or not stablized:
            if iter >= _ITERCAP:
                print('Exceeded', _ITERCAP, 'iterations!')
                break

            if error <= self.errorBound: stablized = True
            else: stabalized = False

            #Get the next iteration
            wPrev = copy.deepcopy(w)
            w -= self.stepsize * self.getGradient(w, self.x_train, self.y_train)
            error = np.linalg.norm(wPrev - w)

            #Record iterations
            iter += 1
            ROUND.append(iter)
            loss = self.getLoss(w, self.x_train, self.y_train)
            LOSS.append(loss)

        if self.plotGD:
            plt.style.use('bmh')
            plt.plot(ROUND, LOSS, label = 'Learning rate = %f'%(self.learningRate), color=_COLORS[colorIndex])
            plt.legend()
            plt.title('Least Square Error vs. Gradient Descent Iteration \n Optimal found at iteration %d with a loss of %1.4f' %(iter, loss), loc='center', size=12)
            # plt.show()
        return w


    def getSGDWeight(self):
        '''
        Get the exact gradient descent weight, full-batch
        Lerning rate is set to be a constant here
        '''
        w = np.zeros((self.num_dimension, 1))
        error, stablized, iter, ROUND, loss, lossMin, LOSS = float('inf'), 0, 0, [], float('inf'), float('inf'), []

        while stablized <= 1:
            if iter >= _ITERCAP:
                print('Exceeded', _ITERCAP, 'iterations!')
                break

            #Convergence conditions
            if loss >= lossMin:
                stablized += 0.1
            elif error <= self.errorBound:
                stablized += 0.1
                self.stepsize /= 2
            else:
                lossMin = copy.deepcopy(loss)

            #Get the next iteration
            error = 0
            for i in range(0, self.num_train, self.batchSize):
                wPrev = copy.deepcopy(w)
                X, Y = self.x_train_Random[i:i+self.batchSize], self.y_train_Random[i:i+self.batchSize]
                w -= self.stepsize * self.getGradient(w, X, Y)
                error += np.linalg.norm(wPrev - w)/self.num_batches

            #Record iterations
            iter += 1
            ROUND.append(iter)
            loss = self.getLoss(w, self.x_train, self.y_train)
            LOSS.append(loss)

        if self.plotGD:
            plt.style.use('bmh')
            plt.plot(ROUND, LOSS, label = 'Learning rate = %f'%(self.learningRate), color=_COLORS[colorIndex])
            plt.legend()
            plt.title('Least Square Error vs. Gradient Descent Iteration \n Optimal found at iteration %d with a loss of %1.4f' %(iter, loss), loc='center', size=12)
            # plt.show()
        return w


    def getOptimalWeight(self):
        '''
        Get the exact optimal weight using Singular Value Decomposition, full-batch
        '''
        U, s, VT = np.linalg.svd(self.x_train, full_matrices=False) # The economy SVD (if want full SVD, change the second parameter to True)
        S = np.zeros((U.shape[1], VT.shape[0]))
        # print('Dimensions:', U.shape, S.shape, VT.shape, '\nSVD Composition:', U.dot(S.dot(VT)))
        S[:VT.shape[0], :VT.shape[0]] = np.diag(s)
        w = VT.T.dot(np.linalg.inv(S)).dot(U.T).dot(self.y_train)
        return w


    def runRegression(self, method, display=False):
        if method == 'SVD':
            self.w = self.getOptimalWeight()
        elif method == 'GD':
            self.w = self.getGDWeight()
        elif method == 'SGD':
            self.preprocessSGD()
            self.w = self.getSGDWeight()

        x, y_actual = self.x_test, self.y_test
        y_predicted = x.dot(self.w)
        rmse = np.sqrt(pow(np.array(y_predicted-y_actual), 2).mean())
        print('RMSE is', rmse, 'for data', self.dataset, 'with linear regression on the set.')

        if display:
            plt.style.use('bmh')
            plt.scatter(x[:, 1], y_actual[:, 0], s=3, color=_COLORS[2])
            plt.scatter(x[:, 1], y_predicted[:, 0], marker = '*', s=3, color=_COLORS[0])
            plt.legend(('Actual', 'Prediction'))
            # plt.plot(X[:, 0], Y_predicted[:, 0], linewidth=1, color=_COLORS[0])
            plt.title('Linear Regression on the test set of "%s" with %s\n Resulting RMSE = %1.4f' %(self.dataset, method, rmse), loc='center', size=12)
            plt.show()
        return rmse




def Q1_GD():
    GD = LinearRegression()
    global colorIndex
    colorIndex = 0
    for learningRate in [0.001, 0.05, 0.1, 0.5, 1]:
        GD.setParameters(convergenceError=0.0001, learningRate=learningRate, GDvisualization=True)
        GD.runRegression('GD')
        colorIndex += 1
    plt.show()


def Q1_SGD():
    SGD = LinearRegression()
    global colorIndex
    colorIndex = 0
    for learningRate in [0.001, 0.05, 0.1, 0.5, 1]:
        SGD.setParameters(convergenceError=0.0001, learningRate=learningRate, batchSize=1, GDvisualization=True)
        SGD.runRegression('SGD')
        colorIndex += 1
    plt.show()


if __name__ == '__main__':
    Q1_SGD()
