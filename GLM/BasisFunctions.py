import math
import numpy as np
from numpy import dot

class BasisFunctions:
    def __init__(self, model='gaussian', lamb=0, M=3, degree=5):
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
        if self.model == 'gaussian': s = (max(set)-min(set))/(self.M-2)

        for i in range(1, self.degree):
            if self.model == 'polynomial':
                phi_i = pow(x, i)

            if self.model == 'gaussian':
                mu_i = min(set) + (i-1)*s
                phi_i = math.exp(-pow((x-mu_i), 2)/(2*pow(s, 2)))

            row.append(phi_i)
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
