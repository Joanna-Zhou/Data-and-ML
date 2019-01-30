import pandas as pd
import matplotlib.pyplot as plt
from kNNValidation import *

_COLORS = ['#d6616b', '#e6550d', '#fdae6b', '#ffbb78', '#e7ba52', '#dbdb8d']

def plotValidPredictionCurves(datasetName, distanceHeuristic, kRange, modificationIndex):
    '''
    Plot the prediction curve of RMSE(%) vs. k
    '''
    data = {'k':[], 'RMSE':[]}
    for k in kRange:
        data['k'].append(k)
        data['RMSE'].append(RMSELoss_Regression(datasetName, distanceHeuristic, k, modificationIndex))
    rmseOptimal = min(data['RMSE'])
    kOptimalIndex = list(data['RMSE']).index(rmseOptimal)
    kOptimal = data['k'][kOptimalIndex]
    curve = pd.DataFrame(data)
    curve.plot(x='k',y='RMSE',color=_COLORS)
    plt.style.use('bmh')
    plt.xlabel('number of nearest neighbours k')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. k on Dataset "%s" with %s Distance -- Valid Set\nOptimal k is %d with RMSE %1.2f' %(datasetName, distanceHeuristic, kOptimal, rmseOptimal), loc='center', size=12)
    plt.savefig('kNNRegression-ValidCurve-mauna_loa.png')
    return kOptimal


def plotTestPredictionCurves(datasetName, distanceHeuristic, kRange, modificationIndex):
    '''
    Plot the prediction curve of RMSE(%) vs. k
    '''
    data = {'k':[], 'RMSE':[]}
    for k in kRange:
        data['k'].append(k)
        kNNtest = kNNTraining(datasetName, distanceHeuristic, k, modificationIndex)
        kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
        errorList, rmse = [], 0
        if modificationIndex in [1, 2]:
            for i in range(kNNtest.num_testSet):
                kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_test[i], kNNtest.y_test[i], kNNtest.modificationIndex)
                errorList.append(error)
            else: # if k = 3 or 4
                pass
        data['RMSE'].append(np.sqrt(pow(np.array(errorList), 2).mean()))
    rmseOptimal = min(data['RMSE'])
    kOptimalIndex = list(data['RMSE']).index(rmseOptimal)
    kOptimal = data['k'][kOptimalIndex]
    curve = pd.DataFrame(data)
    curve.plot(x='k',y='RMSE', color=_COLORS)
    plt.style.use('bmh')
    plt.xlabel('number of nearest neighbours k')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. k on Dataset "%s" with %s Distance -- Test Set\nOptimal k is %d with RMSE %1.2f' %(datasetName, distanceHeuristic, kOptimal, rmseOptimal), loc='center', size=12)
    plt.savefig('kNNRegression-TestCurve-mauna_loa.png')
    return kOptimal


def plotValidSet_Regression(datasetName, distanceHeuristic, k, modificationIndex):
    '''
    Plot the test set's predicted y-values vs. labelled y-values
    '''
    print(k, 'nearest neighbours using', distanceHeuristic, 'distance:')
    kNNtest = kNNTraining(datasetName, distanceHeuristic, k, modificationIndex)
    kNNtest.foldDataset(1)
    kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
    y, errorList, rmse =[], [], 0
    if modificationIndex in [1, 2]:
        for i in range(kNNtest.num_validSet):
            kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_valid[i], kNNtest.y_valid[i], kNNtest.modificationIndex)
            y.append(kNNValue)
            errorList.append(error)
            # print(kNNtest.y_valid[i], kNNValue)
        else: # if k = 3 or 4
            pass
    rmse = np.sqrt(pow(np.array(errorList), 2).mean())

    data = {'x': list(np.transpose(kNNtest.x_valid)[0]), 'y_Labelled': list(np.transpose(kNNtest.y_valid)[0]), 'y_Predicted': y}
    df = pd.DataFrame(data)
    df.plot(kind='scatter',x='x',y='y_Labelled', ax=plt.gca())
    df.plot(kind='scatter',x='x',y='y_Predicted', ax=plt.gca(), color=_COLORS[k%len(_COLORS)])
    plt.style.use('bmh')
    plt.legend(('y_Labelled', 'y_Predicted'))
    plt.title('Predictions of Dataset "%s"\nwith %s Distance and Optimal k = %d -- Validation Set' %(datasetName, distanceHeuristic, k), loc='center', size=12)
    plt.savefig('kNNRegression-ValidSet-mauna_loa.png')
    return rmse


def plotTestSet_Regression(datasetName, distanceHeuristic, k, modificationIndex):
    '''
    Plot the test set's predicted y-values vs. labelled y-values
    '''
    print(k, 'nearest neighbours using', distanceHeuristic, 'distance:')
    kNNtest = kNNTraining(datasetName, distanceHeuristic, k, modificationIndex)
    kNNtest.x_train, kNNtest.y_train = kNNtest.x_all, kNNtest.y_all
    y, errorList, rmse =[], [], 0
    if modificationIndex in [1, 2]:
        for i in range(kNNtest.num_testSet):
            kNNValue, error, correctness = kNNtest.kNNRegression(kNNtest.x_test[i], kNNtest.y_test[i], kNNtest.modificationIndex)
            y.append(kNNValue)
            errorList.append(error)
            # print(kNNtest.y_test[i], kNNValue)
        else: # if k = 3 or 4
            pass
    rmse = np.sqrt(pow(np.array(errorList), 2).mean())

    data = {'x': list(np.transpose(kNNtest.x_test)[0]), 'y_Labelled': list(np.transpose(kNNtest.y_test)[0]), 'y_Predicted': y}
    df = pd.DataFrame(data)
    df.plot(kind='scatter',x='x',y='y_Labelled',ax=plt.gca())
    df.plot(kind='scatter',x='x',y='y_Predicted', ax=plt.gca(), color=_COLORS[k%len(_COLORS)])
    plt.style.use('bmh')
    plt.legend(('y_Labelled', 'y_Predicted'))
    plt.title('Predictions of Dataset "%s"\nwith %s Distance and Optimal k = %d -- Test Set' %(datasetName, distanceHeuristic, k), loc='center', size=12)
    plt.savefig('kNNRegression-TestSet-mauna_loa.png')


if __name__ == '__main__':
    print(plotValidPredictionCurves('pumadyn32nm', 'l2', range(1, 31), 3))
    # print(plotTestPredictionCurves('mauna_loa', 'l2', range(1, 41), 1))
    # for k in range(3, 10):
    #     plotTestSet_Regression('mauna_loa', 'l2', k, 2)
    # for k in range(2, 9):
    #     plotValidSet_Regression('mauna_loa', 'l2', k, 2)
