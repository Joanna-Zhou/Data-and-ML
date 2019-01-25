from kNNPreprocessing import *

def kNN_classifier(xy_train, xy, num_dimension, num_classes, num_trainSet, method = 'l2', k = 1):
    y_actual = xy[num_dimension:] # Extract y from the pair of data, later used to compared with the test result
    iNN = getNeighbours(xy_train, xy, num_dimension, num_classes, num_trainSet, method, k)
    vote, count = np.unique(y train[i nn], return counts=True)
    y_test = vote[np.argmax(count)]



def getNeighbours(xy_train, xy, num_dimension, num_classes, num_trainSet, method = 'l2', k = 1):
    '''
    Get k nearest neighbours for a given x
    INPUT: xy_train: the training set, must be a N-by-(D+#Classes) matrix
    INOUT: xy: a 1-dimensional vector, typically a row from xy_test or xy_valid
    INPUT: num_dimension, num_classes, num_trainSet: parameters directly returned by function loadData(datasetName)
    INPUT: method: distance calculation method, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a list of arrays in xy_train that are the k nearest neighbours of x
    '''
    distances = [getDistance(xy_train[i], xy, num_dimension, method) for i in range(0, num_trainSet)]
    iNN = np.argpartition(distances, range(k))[:k]
    return iNN, distances[iNN]

def getDistance(xy1, xy2, xdimension, method = 'l2'):
    '''
    Calculates the distance with specified method (default is 'l2')
    INPUT: xy1 and xy2: 1-dimensional vectors (two rows in a dataset)
    INPUT: method: 'l1', 'l2', 'linf'
    OUTPUT: a numeric value of the distance
    '''
    try:
        x1, x2 = xy1[: xdimension], xy2[: xdimension] # Extract x1, ..., xD
        # print('Class of x1:',xy1[xdimension:], '\nClass of x2:', xy2[xdimension:])
        sum_distance = 0 # Initiate the distance
        if method == 'l1':
            for i in range(0, xdimension):
                sum_distance += abs(x1[i] - x2[i])
            return sum_distance
        elif method == 'l2':
            for i in range(0, xdimension):
                sum_distance += pow((x1[i] - x2[i]), 2)
            return math.sqrt(sum_distance)
        elif method == 'linf':
            return max(abs(x1 - x2))
        else:
            print("Error! Input 'method' must be one of 'l1', 'l2', and 'linf'.")
    except:
        print("Error! xy1 and xy2 must be 1-dimensional vectors.")
        print("x1 is now a", type(x1), 'in shape', np.shape(x1))



if __name__ == '__main__':
    xy_train, xy_valid, xy_test, num_dimension, num_classes, num_trainSet = loadData('mnist_small')
    # printData(xy_test[0])
    # print(getDistance(xy_test[1], xy_test[29], num_dimension, 'l2'))
    print(getNeighbours(xy_train, xy_test[1], num_dimension, num_classes, num_trainSet, 'l2', k = 5)[1])
