import math
import numpy as np
from numpy import dot
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#e7ba52', '#dbdb8d']


class Kernels:
    def __init__(self, M=3, model='gaussian', lamb=0, theta=0.1, degree=2):
        self.M = M
        self.model = model
        self.lamb = lamb
        self.theta = theta # only if gaussian kernel is used
        self.degree = degree # only if polynomial kernel is used


    def getGram_Gaussian(self, set1, set2):
        '''
        Calculates the Gram matrix constructed by gaussian kernel k(x,z) = exp(-abs(x-z)^2/theta)
        INPUT: set1 and set2 are N-by-D and N'-by-D matrices
        '''
        # K = np.exp(-np.square(set1-np.transpose(set2))/self.theta) # For D=1
        # pairwise_dists = squareform(pdist(set1, 'euclidean'))
        pairwise_dists = cdist(set1, set2)
        K = np.exp(-pairwise_dists**2 / self.theta)
        return K


    def getGram_Linear(self, set1, set2):
        '''
        Calculates the Gram matrix constructed by linear kernel k(x,z) = x.T * z
        INPUT: set1 and se2 are N-by-D and N'-by-D matrices
        '''
        K = set1.dot(set2.T)
        return K


    def getGram_Polynomial(self, set1, set2):
        '''
        Calculates the Gram matrix constructed by linear kernel k(x,z) = (x.T * z + 1)^d
        INPUT: set1 and set2 are N-by-D and N'-by-D matrices
        '''
        N, M = set1.shape[0], set2.shape[0]
        base = set1.dot(set2.T) + 1 #np.ones((N, M))
        K = np.power(base, self.degree)
        return K


    def getGram_Sinusoid(self, set1, set2):
        '''
        Calculates the Gram matrix constructed by gaussian kernel k(x,z) = cos(wx)*cos(wz) + sin(wx)*sin(wz) = cos(w(x-z))
        INPUT: set1 and set2 are N-by-D and N'-by-D matrices
        '''
        period = 0.057
        K = np.cos((set1-set2.T)*2*math.pi/period)
        return K


    def getGram_DIY(self, set1, set2):
        '''
        Combines polynomial kernel and sinusoidal kernel, to translate the DIY basis function
        INPUT: set1 and set2 are N-by-D and N'-by-D matrices
        '''
        return self.getGram_Polynomial(set1, set2) + self.getGram_Sinusoid(set1, set2)


    def getGram(self, set1, set2):
        if self.model == 'gaussian':
            K = self.getGram_Gaussian(set1, set2)
        elif self.model == 'polynomial':
            K = self.getGram_Polynomial(set1, set2)
        elif self.model == 'DIY':
            K = self.getGram_DIY(set1, set2)
        # print('Gram Matrix:\n', K)
        return K


    def getAlpha(self, x_train, y_train):
        '''
        Uses the Cholesky factorization method to compute matrix alpha for f(X, w) = K * alpha
        MUST call getGram before any testing, since self.K is defined in that function
        OUTPUT: alpha = inverse(K + lamb*I) * y_train
        '''
        self.K = self.getGram(x_train, x_train)
        try:
            R = cho_factor(self.K + np.identity(self.K.shape[0]) * self.lamb)
        except np.linalg.linalg.LinAlgError:
            print('\nNot Positive Definite! Eigenvalues:\n', np.linalg.eigh(self.K)[0])
        # alpha = np.linalg.inv(R).dot(R.T).dot(self.K.T).dot(y_train)
        alpha = cho_solve(R, y_train)
        # print(alpha)
        self.alpha = alpha


    def kernelVisualization(self):
        '''
        Plot k(0, z) and k(1, z+1) where z ranges from -0.1 to 1
        '''
        Z = np.array([[z/1000] for z in range(-100, 101)])
        K0 = self.getGram(np.array([0]), Z)
        K1 = self.getGram(np.array([1]), (Z+1))

        plt.style.use('bmh')
        plt.plot(Z[:, 0], K0.T, linewidth=1, color=_COLORS[0])
        plt.scatter(Z[:, 0], K1.T, linewidth=1, color=_COLORS[2])
        plt.legend(('k(0, z)', 'k(1, z+1)'))
        # plt.plot(X[:, 0], Y_predicted[:, 0], linewidth=1, color=_COLORS[0])
        plt.title('Kernel Visualization -- %s'%(self.model), loc='center', size=12)
        plt.show()
#
# import numpy as np
# x_train = np.array([[1], [2], [3]])
# zz_train = np.array([[3], [2], [2]])
# z_train = np.array([[1], [2]])
# print(np.array(x_train-z_train.T))
# x_train = np.array([[1], [2], [3]])
# method = Kernels(M=3, model='gaussian', lamb=0.005, theta=0.0001, degree=10)
# method.kernelVisualization()
# method.getGram(x_train, x_train)
# method.getAlpha(x_train, x_train)
# print(np.linalg.eigh(method.K)[0])
