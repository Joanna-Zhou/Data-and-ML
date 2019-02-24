import math
import numpy as np
from numpy import dot

class BasisFunctions:
    def __init__(self, x_train, model='gaussian', lamb=0, M=3, degree=4):
        self.x_train = x_train
        self.model = model
        self.lamb = lamb
        self.M = M
        self.degree = degree

    def getPhi(self, x, set):
        '''
        Returns a row vector [phi_0(x) phi_1(x) ... phi_{M-1}(x)]
        INPUT x: a column vector of length = self.dimension. In this case of mauna_loa, just a scalar
        INPUT M: dimension of the output row vector
        INPUT: model: basis function models, such as 'polynomial', 'gaussian', 'fourier'
        '''
        row = [1] # phi_0 is always 1
        if self.model == 'gaussian':
            s = (max(self.x_train)-min(self.x_train))/(self.M-2)
            for i in range(1, self.M):
                mu_i = min(self.x_train) + (i-1)*s
                phi_i = math.exp(-pow((x-mu_i), 2)/(2*pow(s, 2)))
                row.append(phi_i)

        if self.model == 'polynomial':
            for i in range(1, self.degree):
                phi_i = pow(x, i)
                row.append(phi_i)

        if self.model == 'fourier':
            period = math.floor((self.M-1)/2)
            omega = 2*math.pi/period
            for i in range(1, period):
                phi_i_cos, phi_i_sin = math.cos(omega*i*x), math.sin(omega*i*x)
                row.append(phi_i_cos)
                row.append(phi_i_sin)

        if self.model == 'DIY':
            period = 0.057
            for d in range(1, self.degree+1):
                row.append(x**d)
            row.append(-math.sin(x*2*math.pi/period))
            row.append(-math.cos(x*2*math.pi/period))
        return row


    def getPhiMatrix(self, set):
        return np.array([self.getPhi(x, set) for x in set])


    def getWeight(self, x_train, y_train):
        '''
        Uses the economic SVD method to compute vector w for f(X, w) = phiMatrix * w
        MUST call getWeightBasisFunc before any testing, since self.w is defined in that function
        OUTPUT: w = V * inverse(ST*S + lamb*I) * ST * UT * y_train
        '''
        self.phiMatrix = self.getPhiMatrix(x_train)
        U, s, VT = np.linalg.svd(self.phiMatrix, full_matrices=False) # The economy SVD (if want full SVD, change the second parameter to True)
        S = np.zeros((U.shape[1], VT.shape[0]))
        S[:VT.shape[0], :VT.shape[0]] = np.diag(s)
        ST, V, UT = S.T, VT.T, U.T
        lambI = np.identity(U.shape[1]) * self.lamb

        w = V.dot(np.linalg.inv(ST.dot(S)+lambI)).dot(ST).dot(UT).dot(y_train)
        self.w = w


# x_train = np.array([[1], [3], [4]])
# # print(math.cos(2*2*math.pi/4))
# print(x_train)
# # x_train = np.array([[1], [2], [3]])
# method = BasisFunctions(x_train, 'fourier', M=10)
# phiMatrix = method.getPhiMatrix(x_train)
# print(phiMatrix)
