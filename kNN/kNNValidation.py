from kNNTraining import *

def RMSELoss_Regression(datasetName, distanceHeuristic='l2', k=3, modificationIndex=1):
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
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return np.mean(rmseValues)


def RMSELoss_Classification(datasetName, distanceHeuristic, k, modificationIndex=1):
    '''
    Takes a dataset and calculates the root-mean-square-error(RMSE) loss of the regression with specified distance metric amd k value
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: distanceHeuristic: distance calculation distanceHeuristic, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a value of the RMSE loss
    '''
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(k, 'nearest neighbours using', distanceHeuristic, 'distance:')
    rmseValues = [] # 5 rmse values from each fold
    for foldIndex in range(1,6):
        errorList = []
        kNNtest = kNNTraining(datasetName, distanceHeuristic, k, foldIndex, modificationIndex)
        print('Done loading' )
        for i in range(kNNtest.num_testSet): #kNNtest.num_testSet
            kNNValue, correctness = kNNtest.kNNClassification(kNNtest.x_valid[i], kNNtest.y_valid[i])
            errorList.append(correctness == False)
        rmse = np.sqrt(np.array(errorList).mean()*100)
        print('RMSE is', rmse, '%', 'for fold', foldIndex)
        rmseValues.append(rmse)
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return np.array(rmseValues).mean()


if __name__ == '__main__':
    # '''
    RMSELoss_Regression('mauna_loa', 'l2', 2, 3)
    # RMSELoss_Classification('iris', 'l2', 3)
