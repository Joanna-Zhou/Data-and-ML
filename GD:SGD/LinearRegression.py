from data_utils import load_dataset
import numpy as np
import scipy as sp
import pandas as pd
from numpy import dot
import matplotlib.pyplot as plt

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#e7ba52', '#dbdb8d']

class LinearRegression:
    def __init__(self):
        self.dataset = 'pumadyn32nm'
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test  = load_dataset(self.dataset)

        # Extract the first 1000 data points
        self.x_train, self.y_train = self.x_train[:1000], self.y_train[:1000]
        self.num_train = 1000
        (self.num_test, self.num_dimension) = self.x_test.shape

        # Add x0 = 1 to each x vector
        x0 = np.ones((np.shape(self.x_train)[0], 1))
        self.x_train = np.concatenate((x0, self.x_train), axis=1)
        x0 = np.ones((np.shape(self.x_test)[0], 1))
        self.x_test = np.concatenate((x0, self.x_test), axis=1)


    def setParameters(self):
        pass

    def brutalForce(self):
        b = inv(self.x_train.T.dot(self.x_train)).dot(self.x_train.T).dot(self.y_train)
        print(b)
        yhat = self.x_train.dot(b)
        # plot data and predictions
        pyplot.scatter(self.x_train, self.y_train)
        pyplot.plot(self.x_train, yhat, color='red')
        pyplot.show()


    def exactOptimalWeight(self):
        '''
        '''
        U, s, VT = np.linalg.svd(self.x_train, full_matrices=False) # The economy SVD (if want full SVD, change the second parameter to True)
        S = np.zeros((U.shape[1], VT.shape[0]))
        # print('Dimensions:', U.shape, S.shape, VT.shape, '\nSVD Composition:', U.dot(S.dot(VT)))
        S[:VT.shape[0], :VT.shape[0]] = np.diag(s)
        w = VT.T.dot(np.linalg.inv(S)).dot(U.T).dot(self.y_train)
        return w



    def runRegression(self, method, display='off'):
        if method == 'SVD':
            self.w = self.exactOptimalWeight()

        x, y_actual = self.x_test, self.y_test
        y_predicted = x.dot(self.w)
        rmse = np.sqrt(pow(np.array(y_predicted-y_actual), 2).mean())
        print('RMSE is', rmse, 'for data', self.dataset, 'with linear regression on the', set, 'set.')

        if display == 'on':
            plt.style.use('bmh')
            plt.scatter(x[:, 1], y_actual[:, 0], s=3, color=_COLORS[2])
            plt.scatter(x[:, 1], y_predicted[:, 0], marker = '*', s=3, color=_COLORS[0])
            plt.legend(('Actual', 'Prediction'))
            # plt.plot(X[:, 0], Y_predicted[:, 0], linewidth=1, color=_COLORS[0])
            plt.title('Linear Regression on the test set of "%s" with %s\n Resulting RMSE = %1.4f' %(self.dataset, method, rmse), loc='center', size=12)
            plt.show()
        return rmse



if __name__ == '__main__':
    LRtest = LinearRegression()
    LRtest.runRegression('SVD', 'on')
