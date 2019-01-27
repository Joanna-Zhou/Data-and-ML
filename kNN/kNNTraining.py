from kNNPreprocessing import *

class kNNTraining:
    def __init__(self, datasetName, method, k, foldIndex):
        '''
        To run kNNTraining, please declare a class with the desired parameters and then call
        "kNNtest1.kNNRegression(kNNtest1.x_test[i], kNNtest1.y_test[i])" in a loop pf desired i
        '''
        self.method = method # distance calculation method, 'l1', 'l2', or 'linf'
        self.k = k  # number of nearest neighbours required

        # Extraxt datasets associated with the dataset's name
        # x/y_train: the training sets, must be a N-by-D matrix for x_train and N-by-(#Class) for y_train
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = loadData(datasetName, foldIndex)
        # self.xy_train, self.xy_valid, self.xy_test, self.num_dimension, self.num_classes, self.num_trainSet = concatenate(self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test)
        self.num_dimension = np.shape(self.x_test)[1]
        self.num_trainSet = np.shape(self.x_train)[0]
        self.num_validSet = np.shape(self.x_valid)[0]
        self.num_testSet = np.shape(self.x_test)[0]

    def kNNClassification(self, x, y):
        '''
        Classify which class this x is in and compare to its actual value
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        OUTPUT: kNNClass: a classification result of class y
                correctness: a boolean indicating if the prediction is the same as label
        '''
        actualClass = y
        iNN = self.getNeighbours(x, y)
        # print('Selected', self.k, "nearest neighbours' classes:\n", self.y_train[iNN])

        vote, count = np.unique(self.y_train[iNN], axis=0, return_counts=True) # Find the class holding majority
        kNNClass = vote[np.argmax(count)]
        # print('Classified in class', list(kNNClass).index(True), 'and it is actually in class', list(y).index(True))

        correctness = np.unique(kNNClass == actualClass)[0] # Compare to the actual class
        # print('Classified in class', list(kNNClass).index(True), '\nResult is', correctness)
        return kNNClass, correctness


    def kNNRegression(self, x, y):
        '''
        Predict the output value of given x and compare to its actual label y
        INOUT: x, y: 1-dimensional vectors, typically a row from x/y_test or x/y_valid
        OUTPUT: kNNClass: a classification result of class y
                error: error as a percentage
                correctness: a boolean indicating if the prediction is within a certain boundary of its label
        '''
        actualValue = y[0]
        iNN = self.getNeighbours(x, y)
        yNN = self.y_train[iNN]
        # print('Selected', self.k, "nearest neighbours' values:\n", yNN)

        kNNValue = (sum(yNN)/len(yNN))[0]
        # print('Classified in class', list(kNNClass).index(True), 'and it is actually in class', list(y).index(True))

        error = kNNValue - actualValue # Compare to the actual class
        percent_error =  abs(error/actualValue) # Compare to the actual class
        correctness = (percent_error < 0.25)
        # print('Predicted value is', kNNValue, '\nError is', error*100, '%', 'and considered', correctness)
        return kNNValue, error, correctness


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
        Calculates the distance with specified method (default is 'l2')
        INPUT: xy1 and xy2: 1-dimensional vectors (two rows in a dataset)
        INPUT: method: 'l1', 'l2', 'linf'
        OUTPUT: a numeric value of the distance
        '''
        try:
            # print('Label of x1:', y1, '\nLabel of x2:', y2)
            sum_distance = 0 # Initiate the distance
            if self.method == 'l1':
                for i in range(0, self.num_dimension):
                    sum_distance += abs(x1[i] - x2[i])
                return sum_distance
            elif self.method == 'l2':
                for i in range(0, self.num_dimension):
                    sum_distance += pow((x1[i] - x2[i]), 2)
                return math.sqrt(sum_distance)
            elif self.method == 'linf':
                return max(abs(x1 - x2))
            else:
                print("Error! Input 'method' must be one of 'l1', 'l2', and 'linf'.")
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
