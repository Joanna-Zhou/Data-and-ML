from kNNTraining import *
import pandas as pd

def RMSEComparison(datasetName, model, set, kRange, modificationIndex=3):
    '''
    Record the performance (RMSE) on each k value and distance metric
    INPUT: model: one of 'classification' or 'regression'
    INPUT: set: must be one of 'validation' or 'test'
    INPUT: kRange: a range of k to be tested on
    '''
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    dict = {} # with keys: k and distance distanceHeuristic
    optimal = {} # with key: distance distanceHeuristic
    for distanceHeuristic in ['l1', 'l2', 'linf']:
        dict[distanceHeuristic] = {}
        for k in kRange:
            if model == 'regression': dict[distanceHeuristic][k] = RMSELoss_Regression(datasetName, set, distanceHeuristic, k, modificationIndex)
            else: dict[distanceHeuristic][k] = RMSELoss_Classification(datasetName, set, distanceHeuristic, k)
        rmseOptimal = min(dict[distanceHeuristic].values())
        kOptimal = [k for k, rmse in dict[distanceHeuristic].items() if rmse == rmseOptimal]
        optimal[distanceHeuristic] = kOptimal
    print('For dataset '+datasetName+', the test results are:', dict, '\nThe optimal k for each distance heuristic is:', optimal)
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return pd.DataFrame(dict), pd.DataFrame(optimal)


def RMSELoss_Regression(datasetName, set, distanceHeuristic='l2', k=3, modificationIndex=2):
    '''
    Takes a dataset and calculates the root-mean-square-error(RMSE) loss of the regression with specified distance metric amd k value
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: distanceHeuristic: distance calculation distanceHeuristic, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a value of the average RMSE loss across 5 folds
    '''
    np.random.seed(20192019)

    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(k, 'nearest neighbours using', distanceHeuristic, 'distance:')
    kNNtest = kNNTraining(datasetName, distanceHeuristic, k, modificationIndex)
    rmseValues = [] # 5 rmse values from each fold
    if set == 'validation':
        for foldIndex in range(1,6):
            kNNtest.foldDataset(foldIndex)
            errorList = []
            if modificationIndex in [1, 2]:
                for i in range(kNNtest.num_validSet):
                    kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_valid[i], kNNtest.y_valid[i], kNNtest.modificationIndex)
                    errorList.append(error)
                    # print(kNNtest.y_valid[i], kNNValue)
            elif modificationIndex == 3:
                kNNValue, errorList = kNNtest.kNNRegression_3(kNNtest.x_valid, kNNtest.y_valid)
            rmse = np.sqrt(pow(np.array(errorList), 2).mean())
            print('RMSE is', rmse, 'for fold', foldIndex)
            rmseValues.append(rmse)
    else:
        kNNtest.foldDataset(1)
        kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
        errorList = []
        if modificationIndex in [1, 2]:
            for i in range(kNNtest.num_testSet):
                kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_test[i], kNNtest.y_test[i], kNNtest.modificationIndex)
                errorList.append(error)
                # print(kNNtest.y_valid[i], kNNValue)
        elif modificationIndex == 3:
            kNNValue, errorList = kNNtest.kNNRegression_3(kNNtest.x_test, kNNtest.y_test)
        rmseValues.append(np.sqrt(pow(np.array(errorList), 2).mean()))
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return np.mean(rmseValues)


def RMSELoss_Classification(datasetName, set, distanceHeuristic, k):
    '''
    Takes a dataset and calculates the root-mean-square-error(RMSE) loss of the regression with specified distance metric amd k value
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: 'set': must be one of 'validation' or 'test'
    INPUT: distanceHeuristic: distance calculation distanceHeuristic, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a value of the RMSE loss
    '''
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(k, 'nearest neighbours using', distanceHeuristic, 'distance:')
    kNNtest = kNNTraining(datasetName, distanceHeuristic, k)
    rmseValues = [] # 5 rmse values from each fold
    if set == 'validation':
        for foldIndex in range(1,6):
            kNNtest.foldDataset(foldIndex)
            errorList = []
            for i in range(kNNtest.num_validSet):
                kNNValue, correctness = kNNtest.kNNClassification(kNNtest.x_valid[i], kNNtest.y_valid[i])
                errorList.append(not correctness)
            rmse = np.sqrt(np.array(errorList).mean())
            print('RMSE is', rmse, 'for fold', foldIndex)
            rmseValues.append(rmse)
    else:
        kNNtest.foldDataset(1)
        kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
        errorList = []
        for i in range(kNNtest.num_testSet):
            kNNValue, correctness = kNNtest.kNNClassification(kNNtest.x_test[i], kNNtest.y_test[i])
            errorList.append(not correctness)
        rmseValues.append(np.sqrt(np.array(errorList).mean()))
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return np.mean(rmseValues)


if __name__ == '__main__':
    # '''
    # RMSELoss_Regression('mauna_loa', 'l2', 2, 3)
    dict_all = {}
    # dict_all['mauna_loa'] = RMSEComparison('mauna_loa', 'regression', 'test', range(1, 11))
    # dict_all['pumadyn32nm'] = RMSEComparison('pumadyn32nm', 'regression', range(20, 30))
    # dict_all['rosenbrock'] = RMSEComparison('rosenbrock', 'regression', range(1, 11))
    # print(dict_all)
    # RMSELoss_Classification('iris', 'l2', 3)

    # dict_all = {}
    dict_all['iris'] = RMSEComparison('iris', 'classification', 'test', range(1, 21))
    dict_all['mnist_small'] = RMSEComparison('mnist_small', 'classification', 'test', range(1, 6))
    print(dict_all)
