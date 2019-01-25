from kNNPreprocessing import *


def distance(xy1, xy2, xdimension, method = 'l2'):
    '''
    Calculates the distance with specified method (default is 'l2')
    INPUT: xy1 and xy2: 1-dimensional vectors (two rows in a dataset)
    INPUT: method: 'l1', 'l2', 'linf'
    OUTPUT: a numeric value of the distance
    '''
    try:
        x1, x2 = xy1[: xdimension], xy2[: xdimension] # Extract x1, ..., xD
        print('Class of x1:',xy1[xdimension:], '\nClass of x2:', xy2[xdimension:])
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
    xy_train, xy_valid, xy_test, num_dimension, num_classes = loadData('mnist_small')
    # printData(xy_test[0])
    print(distance(xy_test[1], xy_test[29], num_dimension, 'l2'))
