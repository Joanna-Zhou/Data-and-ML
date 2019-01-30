from kNNTraining import *
import matplotlib.pyplot as plt
import pandas as pd
import os

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#ffbb78', '#e7ba52', '#dbdb8d']

def kNNPerformance(datasetName, modificationRange, distanceHeuristic='l2', k=5, dRange=[2, 10, 20, 50, 100]):
    '''
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: modificationRange: a range of modification indices corresponding to a, b, c and/or d in Q3 of the assignment
    INPUT: distanceHeuristic: distance calculation distanceHeuristic, here we use 'l2'
    INPUT: k: number of nearest neighbours required, here we use 5
    OUTPUT: a value of the average RMSE loss across 5 folds
    '''
    for d in dRange:
        kNNtest = kNNTraining(datasetName, distanceHeuristic, k, 1, d)
        runtimes = []
        for modificationIndex in modificationRange:
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print('5 nearest neighbours using l2 distance with method', modificationIndex)
            kNNtest.modificationIndex = modificationIndex
            tic = os.times()[0] # Record starting time
            kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
            y, errorList, rmse =[], [], 0
            if modificationIndex in [1, 2]:
                for i in range(kNNtest.num_testSet):
                    kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_test[i], kNNtest.y_test[i], kNNtest.modificationIndex)
                    y.append(kNNValue)
                    errorList.append(error)
            elif modificationIndex == 3:
                kNNValue, errorList = kNNtest.kNNRegression_3(kNNtest.x_test, kNNtest.y_test)
            runtimes.append(os.times()[0] - tic)
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        plt.style.use('bmh') # plt.style.use('ggplot')
        xAxis = np.arange(len(modificationRange))
        plt.bar(xAxis, height= runtimes, alpha=0.3, color=_COLORS)
        for x in xAxis:
            plt.text(x=x-0.3, y=runtimes[x]+0.001, s='d=%d: %1.2fs'%(d, runtimes[x]), size = 7)
    plt.xticks(xAxis, ['Double For-loops', 'Single For-loop', 'Full Vectorization', 'K-d Tree'][:len(modificationRange)], size = 7)
    plt.xlabel('Modification Method')
    plt.ylabel('Runtime [sec]')
    # plt.legend(['Double For-loops', 'Single For-loop', 'Full Vectorization', 'K-d Tree'][:len(modificationRange)], loc='upper right')
    plt.title('Runtime Comparison between Different Distance Calculation Codes \non Dataset "%s" with %s Distance k = %d' %(datasetName, distanceHeuristic, k), loc='center', size = 12)
    plt.savefig('kNNPerformance-TestSet-rosenbrock.png')
    return runtimes


if __name__ == '__main__':

    runtimes = kNNPerformance('rosenbrock', modificationRange=range(1,4), dRange=[2, 100]) #rosenbrock
