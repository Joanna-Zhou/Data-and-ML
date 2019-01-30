from kNNPreprocessing import *

class kNNTraining:
    def __init__(self, datasetName, distanceHeuristic='l2', k=3, modificationIndex=1):
        '''
        To run kNNTraining, please declare a class with the desired parameters and then call
        "kNNtest1.kNNRegression(kNNtest1.x_test[i], kNNtest1.y_test[i])" in a loop pf desired i
        '''
        self.distanceHeuristic = distanceHeuristic # distance calculation distanceHeuristic, 'l1', 'l2', or 'linf'
        self.k = k  # number of nearest neighbours required
        self.modificationIndex = modificationIndex # chooses which method to use to modify regression, corresponding to Q3

        # Extraxt datasets associated with the dataset's name
        # x/y_train: the training sets, must be a N-by-D matrix for x_train and N-by-(#Class) for y_train
        self.index_all, self.x_all, self.x_test, self.y_all, self.y_test = loadData(datasetName)
        self.num_dimension = np.shape(self.x_test)[1]
        self.num_validSet = round(np.shape(self.x_all)[0]/5)
        self.num_trainSet = np.shape(self.x_all)[0] - self.num_validSet
        self.num_testSet = np.shape(self.x_test)[0]

    def foldDataset(self, foldIndex):
        self.x_train, self.x_valid, self.y_train, self.y_valid = foldDataset(self.index_all, self.x_all, self.y_all, foldIndex)


    def kNNClassification(self, x, y):
        '''
        Classify which class this x is in and compare to its actual value
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        OUTPUT: kNNClass: a classification result of class y
                correctness: a boolean indicating if the prediction is the same as label
        '''
        actualClass = y
        iNN = self.getNeighbours_2(x, y)
        # print('Selected', self.k, "nearest neighbours' classes:\n", self.y_train[iNN])

        vote, count = np.unique(self.y_train[iNN], axis=0, return_counts=True) # Find the class holding majority
        kNNClass = vote[np.argmax(count)]
        # print('Classified in class', list(kNNClass).index(True), 'and it is actually in class', list(y).index(True))

        correctness = np.unique(kNNClass == actualClass)[0] # Compare to the actual class
        # print('Classified in class', list(kNNClass).index(True), '\nResult is', correctness)
        return kNNClass, correctness


    def kNNRegression(self, x, y, modificationIndex):
        '''
        Predict the output value of given x and compare to its actual label y
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        INPUT: modificationIndex: one of 1, 2, 3
        OUTPUT: kNNClass: a classification result of class y
                error: absolute difference between predicted and given y's
                correctness: a boolean indicating if the prediction is within a certain boundary of its label
        '''
        actualValue = y[0]
        if modificationIndex == 1: iNN = self.getNeighbours(x, y)
        elif modificationIndex == 2: iNN = self.getNeighbours_2(x, y)
        yNN = self.y_train[iNN]
        # print('Selected', self.k, "nearest neighbours' values:\n", yNN)

        kNNValue = (sum(yNN)/len(yNN))[0]
        # print('Classified in class', list(kNNClass).index(True), 'and it is actually in class', list(y).index(True))

        error = kNNValue - actualValue # Compare to the actual class
        percent_error =  abs(error/actualValue) # Compare to the actual class
        correctness = (percent_error < 0.25)
        # print('Predicted value is', kNNValue, '\nError is', error*100, '%', 'and considered', correctness)
        return kNNValue, error, correctness


    def kNNRegression_3(self, x_set, y_set):
        '''
        Predict the output values of ALL x's and compare to their actual label y's -- fully vectorized
        INOUT: x_set, y_set: either x_test and y_test or x_valind and y_valid
        OUTPUT: kNNValues: results of class y (dimension = num_testSet or num_validSet)
                errorList: array of error
        '''
        x_train = np.broadcast_to(self.x_train,(len(x_set),)+self.x_train.shape)
        y_train = np.broadcast_to(self.y_train,(len(x_set),)+self.y_train.shape)
        x_set = np.expand_dims(x_set, axis=1)

        distances = np.sqrt(np.sum(np.square(x_train - x_set), axis=2))
        iNN = np.argpartition(distances, range(self.k), axis = 1)
        yNN = [y_train[i][iNN[i][:self.k]] for i in range(len(x_set))]
        kNNValues = [(sum(yNN[i])/len(yNN[i])) for i in range(len(x_set))]

        errorList = kNNValues - y_set # Compare to the actual value
        return kNNValues, errorList # Both should be arrays


    def getNeighbours_2(self, x, y):
        '''
        Get k nearest neighbours for a given x using vectorized python code instead of the for-loop over training points
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        OUTPUT: a list of indexes of data in x_train that are the k nearest neighbours of x
        '''
        distances = np.sqrt(np.sum(np.square(self.x_train - x), axis=1))
        iNN = np.argpartition(distances, range(self.k))[:self.k]
        return iNN


    def getNeighbours(self, x, y):
        '''
        Get k nearest neighbours for a given x
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        OUTPUT: a list of indexes of data in x_train that are the k nearest neighbours of x
        '''
        distances = [self.getDistance(self.x_train[i], self.y_train[i], x, y) for i in range(self.num_trainSet)]
        iNN = np.argpartition(distances, range(self.k))[:self.k]
        return iNN


    def getDistance(self, x1, y1, x2, y2):
        '''
        Calculates the distance with specified distanceHeuristic (default is 'l2')
        INPUT: xy1 and xy2: 1-dimensional vectors (two rows in a dataset)
        INPUT: distanceHeuristic: 'l1', 'l2', 'linf'
        OUTPUT: a numeric value of the distance
        '''
        try:
            # print('Label of x1:', y1, '\nLabel of x2:', y2)
            sum_distance = 0 # Initiate the distance
            if self.distanceHeuristic == 'l1': return np.linalg.norm(x1 - x2, 1)
            elif self.distanceHeuristic == 'l2': return np.linalg.norm(x1 - x2)
            elif self.distanceHeuristic == 'linf': return np.linalg.norm(x1 - x2, 'inf')
            else: print("Error! Input 'distanceHeuristic' must be one of 'l1', 'l2', and 'linf'.")
        except:
            print("Error! xy1 and xy2 must be 1-dimensional vectors.")
            print("x1 is now a", type(x1), 'in shape', np.shape(x1))



if __name__ == '__main__':
    # '''
    kNNtestClass = kNNTraining('iris', 'linf', 3, 5) # iris or mnist_small
    for i in range(10): # This test yielded 9 true predictions and 1 false in 55 seconds
        kNNtestClass.kNNClassification(kNNtestClass.x_test[i], kNNtestClass.y_test[i])
    '''
    kNNtestReg = kNNTraining('rosenbrock', 'l2', 10) # mauna_loa, pumadyn32nm or rosenbrock
    for i in range(0, 5): # This test yielded 9 true predictions and 1 false in 55 seconds
        # kNNtestReg.kNNRegression(kNNtestReg.x_valid[i], kNNtestReg.y_valid[i])
        kNNtestReg.kNNRegression(kNNtestReg.x_train[i], kNNtestReg.y_train[i])
    '''
