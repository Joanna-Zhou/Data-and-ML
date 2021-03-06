from data_utils import load_dataset
import numpy as np
from numpy import dot
from matplotlib import pyplot

class LinearRegression:
    def __init__(self, datasetName, d=2):
        '''
        To run kNNTraining, please declare a class with the desired parameters and then call
        "kNNtest1.kNNRegression(kNNtest1.x_test[i], kNNtest1.y_test[i])" in a loop pf desired i
        '''
        if datasetName == 'rosenbrock':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(datasetName, n_train=1000, d=d)
        else:
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test  = load_dataset(datasetName)

        self.dataset = datasetName

        # Normalizetion of each x data
        mean = self.x_train.mean(axis=0, keepdims=True)
        stddev = self.x_train.std(axis=0, keepdims=True)
        self.x_train = (self.x_train - mean)/stddev
        self.x_valid = (self.x_valid - mean)/stddev
        self.x_test = (self.x_test - mean)/stddev
        self.num_dimension = np.shape(self.x_test)[1]
        self.num_classes = np.shape(self.y_test)[1]

        # Add x0 = 1 to each x vector
        x0 = np.ones((np.shape(self.x_train)[0], 1))
        self.x_train = np.concatenate((x0, self.x_train), axis=1)
        x0 = np.ones((np.shape(self.x_valid)[0], 1))
        self.x_valid = np.concatenate((x0, self.x_valid), axis=1)
        x0 = np.ones((np.shape(self.x_test)[0], 1))
        self.x_test = np.concatenate((x0, self.x_test), axis=1)

        self.w = self.optimalWeight()

    def brutalForce(self):
        b = inv(self.x_train.T.dot(self.x_train)).dot(self.x_train.T).dot(self.y_train)
        print(b)
        yhat = self.x_train.dot(b)
        # plot data and predictions
        pyplot.scatter(self.x_train, self.y_train)
        pyplot.plot(self.x_train, yhat, color='red')
        pyplot.show()


    def optimalWeight(self):
        '''
        '''
        U, s, VT = np.linalg.svd(self.x_train, full_matrices=False) # The economy SVD (if want full SVD, change the second parameter to True)
        S = np.zeros((U.shape[1], VT.shape[0]))
        # print('Dimensions:', U.shape, S.shape, VT.shape, '\nSVD Composition:', U.dot(S.dot(VT)))
        S[:VT.shape[0], :VT.shape[0]] = np.diag(s)
        w = VT.T.dot(np.linalg.inv(S)).dot(U.T).dot(self.y_train)
        return w


    def plotRegression(self, set):
        if set == 'validation': x, y_actual = self.x_valid, self.y_valid
        else: x, y_actual = self.x_test, self.y_test
        y_predicted = x.dot(self.w)
        pyplot.scatter(x[:, 1], y_actual[:, 0])
        pyplot.plot(x[:, 1], y_predicted[:, 0], color='red')
        pyplot.grid()
        pyplot.show()


    def RMSE_Regression(self, set):
        if set == 'validation': x, y_actual = self.x_valid, self.y_valid
        else: x, y_actual = self.x_test, self.y_test
        y_predicted = x.dot(self.w)
        rmse = np.sqrt(pow(np.array(y_predicted-y_actual), 2).mean())
        print('RMSE is', rmse, 'for data', self.dataset, 'with linear regression on the', set, 'set.')
        return rmse


    def RMSE_Classification(self, set):
        if set == 'validation': x, y_actual = self.x_valid, self.y_valid
        else: x, y_actual = self.x_test, self.y_test
        f_predicted = x.dot(self.w)
        y_predicted = []
        for f in f_predicted:
            maxClass = list(f).index(max(f))
            y_base = np.zeros(self.num_classes)
            y_base[maxClass] = 1
            y_predicted.append(y_base)
        rmse = np.sqrt(pow(np.array(np.array(y_predicted)-y_actual), 2).mean())
        print('RMSE is', rmse, 'for data', self.dataset, 'with linear regression on the', set, 'set.')
        return rmse


if __name__ == '__main__':
    # LRtest = LinearRegression('mauna_loa')
    # LRtest.RMSE_Regression('test')
    LRtest = LinearRegression('mnist_small')
    LRtest.RMSE_Classification('test')
