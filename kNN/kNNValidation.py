from kNNTraining import *

def RMSELoss_Regression(datasetName, method, k):
    '''
    Takes a dataset and calculates the root-mean-square-error(RMSE) loss of the regression with specified distance metric amd k value
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: method: distance calculation method, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a value of the average RMSE loss across 5 folds
    '''
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(k, 'nearest neighbours using', method, 'distance:')
    rmseValues = [] # 5 rmse values from each fold
    for foldIndex in range(1,6):
        errorList = []
        kNNtest = kNNTraining(datasetName, method, k, foldIndex)
        for i in range(kNNtest.num_validSet):
            kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_valid[i], kNNtest.y_valid[i])
            errorList.append(error)
            # print(kNNtest.y_valid[i], kNNValue)
        rmse = np.sqrt(pow(np.array(errorList), 2).mean())
        # print('Squared error:', pow(np.array(errorList), 2), 'Max of error:', pow(np.array(errorList), 2).mean())
        print('RMSE is', rmse, 'for fold', foldIndex)
        rmseValues.append(rmse)
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    return np.array(rmseValues).mean()


def RMSELoss_Classification(datasetName, method, k):
    '''
    Takes a dataset and calculates the root-mean-square-error(RMSE) loss of the regression with specified distance metric amd k value
    INPUT: databaseName: must be one of the regression datasets, can't be classification
    INPUT: method: distance calculation method, 'l1', 'l2', or 'linf'
    INPUT: k: number of nearest neighbours required
    OUTPUT: a value of the RMSE loss
    '''
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(k, 'nearest neighbours using', method, 'distance:')
    rmseValues = [] # 5 rmse values from each fold
    for foldIndex in range(1,6):
        errorList = []
        kNNtest = kNNTraining(datasetName, method, k, foldIndex)
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
    RMSELoss_Regression('rosenbrock', 'l2', 7)
    # RMSELoss_Classification('iris', 'l2', 3)
