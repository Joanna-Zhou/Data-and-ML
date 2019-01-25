# import pandas
import math

from data_utils import load_dataset
import numpy as np

def loadData(dataset):
    '''
    INPUT: dataset: a string of the name of file to be loaded. Note that this file must be in the same path as this file
    OUTPUT: Three matrixes of paired dataset, each represented in a Nx(D+1) matrix where each row is [x1, x2, .... xD, y]
    '''
    if dataset == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, n_train=5000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
    train_size, valid_size, test_size = len(x_train),len(x_valid), len(x_test)
    # Pair each x array with its corresponding y array with numpy's concatenate function
    xy_train = np.concatenate([np.transpose(x_train), np.transpose(y_train)]).transpose()
    xy_valid = np.concatenate([np.transpose(x_valid), np.transpose(y_valid)]).transpose()
    xy_test = np.concatenate([np.transpose(x_test), np.transpose(y_test)]).transpose()

    # printData(xy_test)
    return xy_train, xy_valid, xy_test

def printData(dataset, item = 'both'):
    '''
    INPUT: dataset: xy_train, xy_valid, or xy_test -- matrix consist of lists of data pairs
    INPUT: item: 'both', 'data', 'shape' -- what to print
    OUTPUT: print required item
    '''
    try:
        if item == 'both' or item == 'size':
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print(dataset)
        if item == 'both' or item == 'data':
            rows, columns = np.shape(dataset)
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print('This dataset has', rows, 'rows and', columns, 'columns.')
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            # print('Each data pair such as', dataset[0], 'is a type of', type(dataset[0][0]))
    except:
        print("Error! Input 'dataset' must be a 2-dimension matrix.")

    if item != 'both' and item != 'data' and item != 'shape':
        print("Error! Input 'item' must be one of 'both', 'data', 'shape', or default.")


def distance(instance1, instance2, length):

	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)



if __name__ == '__main__':
    loadData('mnist_small')
    # print(str(x_train[2][0])+', '+str(y_train[2][0]))
