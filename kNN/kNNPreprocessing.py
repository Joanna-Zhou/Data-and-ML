# import pandas
import math

from data_utils import load_dataset
import numpy as np

def loadData(datasetName, foldIndex):
    '''
    Loads the dataset and normalize the x_ sets
    INPUT: datasetName: a string of the name of file to be loaded. Note that this file must be in the same path as this file
    INPUT: foldIndex: from 1 to 5, decides how to partition the dataset
    OUTPUT: 6 datasets in array form, 3 of which are normalized x data
    '''
    if datasetName == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(datasetName, n_train=5000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(datasetName)

    x_all = np.concatenate([x_train, x_valid])
    x_train, x_valid = foldDataset(foldIndex, x_all)
    y_all = np.concatenate([y_train, y_valid])
    y_train, y_valid = foldDataset(foldIndex, y_all)

    # Normalizetion of each x data
    mean = x_train.mean(axis=0, keepdims=True)
    stddev = x_train.std(axis=0, keepdims=True)
    x_train = normalization(x_train, mean, stddev)
    x_valed = normalization(x_valid, mean, stddev)
    x_test = normalization(x_test, mean, stddev)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def foldDataset(foldIndex, x_all):
    total = np.shape(x_all)[0]
    oneFifth = round(total/5)
    if foldIndex in [1, 2, 3, 4]:
        x_train = np.concatenate([x_all[:oneFifth*(foldIndex-1)], x_all[oneFifth*foldIndex:]])
        x_test = np.array(x_all[oneFifth*(foldIndex-1):oneFifth*foldIndex])
    elif foldIndex == 5: # for the last fold, cound backwards so that it has the same number of data as the other folds
        x_train = np.array(x_all[:(total-oneFifth)])
        x_test = np.array(x_all[(total-oneFifth):])
    return x_train, x_test

def concatenate(x_train, x_valid, x_test, y_train, y_valid, y_test):
    '''
    Put the datasets in matrix form
    INPUT: 6 datasets from loaddata
    OUTPUT: 3 matrixes of paired dataset, each represented in a N x (D + #Class) matrix where each row is [x1, x2, .... xD, y1, y2, ..., y#Class]
            And the dimension of x^(i) and y^(i), the latter being the number of classes, or 1 if regression
    '''
    # Pair each x array with its corresponding y array with numpy's concatenate function
    # Note that all if Y = {'True', 'False'}, it will become {0, 1} after this step
    xy_train = np.concatenate([np.transpose(x_train), np.transpose(y_train)]).transpose()
    xy_valid = np.concatenate([np.transpose(x_valid), np.transpose(y_valid)]).transpose()
    xy_test = np.concatenate([np.transpose(x_test), np.transpose(y_test)]).transpose()
    num_dimension = np.shape(x_test)[1]
    num_classes = np.shape(y_test)[1]
    num_trainSet = np.shape(x_train)[0]
    return xy_train, xy_valid, xy_test, num_dimension, num_classes, num_trainSet

def normalization(x, mean, stddev):
    '''
    Returned a matrix of x data normalized against x_train's mean and stddev
    '''
    return (x - mean)/stddev


def printData(dataset, item = 'both'):
    '''
    Prints required data and/or shape of specified dataset
    INPUT: dataset: xy_train, xy_valid, or xy_test -- matrix consist of lists of data pairs
    INPUT: item: 'both', 'data', 'shape' -- what to print
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




if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = loadData('iris', 5)
    # xy_train, xy_valid, xy_test, num_dimension, num_classes, num_trainSet = concatenate(x_train, x_valid, x_test, y_train, y_valid, y_test)
    # print (num_dimension, num_classes)
    # printData(xy_test)
