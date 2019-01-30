# import pandas
import math

from data_utils import load_dataset
import numpy as np
import random


def loadData(datasetName, d=2):
    '''
    Loads the dataset and normalize the x_ sets
    INPUT: datasetName: a string of the name of file to be loaded. Note that this file must be in the same path as this file
    OUTPUT: 6 datasets in array form, 3 of which are normalized x data
    '''
    if datasetName == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(datasetName, n_train=1000, d=d)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(datasetName)

    x_all = np.concatenate([x_train, x_valid])
    y_all = np.concatenate([y_train, y_valid])
    index_all = list(range(np.shape(x_all)[0]))
    random.shuffle(index_all)

    # Normalizetion of each x data
    mean = x_all.mean(axis=0, keepdims=True)
    stddev = x_all.std(axis=0, keepdims=True)
    x_all = normalization(x_all, mean, stddev)
    x_test = normalization(x_test, mean, stddev)
    return index_all, x_all, x_test, y_all, y_test


def foldDataset(allIndex, x_all, y_all, foldIndex):
    '''
    Split data into two sets of ratio 4:1 according to the foldIndex
    INPUT: allData: concatenate dataset
    INPUT: foldIndex: from 1 to 5, decides how to partition the dataset (must be from outside the class)
    OUTPUT: train, set: the 4:1 ratio datasets
    '''
    total = len(allIndex)
    oneFifth = round(total/5)
    if foldIndex in [1, 2, 3, 4]:
        index_train = allIndex[:oneFifth*(foldIndex-1)] + allIndex[oneFifth*foldIndex:]
        index_valid = allIndex[oneFifth*(foldIndex-1) : oneFifth*foldIndex]
    elif foldIndex == 5: # for the last fold, cound backwards so that it has the same number of data as the other folds
        index_train = allIndex[:(total-oneFifth)]
        index_valid = allIndex[(total-oneFifth):]
    x_train, x_valid = x_all[index_train], x_all[index_valid]
    y_train, y_valid = y_all[index_train], y_all[index_valid]
    return x_train, x_valid, y_train, y_valid


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
    index_all, x_all, x_test, y_all, y_test = loadData('mauna_loa')
    # xy_train, xy_valid, xy_test, num_dimension, num_classes, num_trainSet = concatenate(x_train, x_valid, x_test, y_train, y_valid, y_test)
    # print (num_dimension, num_classes)
    x_train, x_valid, y_train, y_valid = foldDataset(index_all, x_all, y_all, 5)
    printData(y_valid)
