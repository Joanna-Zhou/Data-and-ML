import math

from data_utils import load_dataset
import numpy as np
from sklearn.neighbors import KDTree

if __name__ == '__main__':
    k = 2
    x_train = np.array([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6], [4, 5, 6, 7, 8]])
    y_train = np.array([[-1], [3], [4]])
    x_test = np.array([[1, 0, 3, 2, 0],[2, 3, 4, 5, 6]])
    y_test = np.array([[0.5], [2.5]])
    # print(x_train, '\n\n', x_test)
    #
    # distances = np.array([np.sqrt(np.sum(np.square(x_train - x_test[0]), axis=1)), np.sqrt(np.sum(np.square(x_train - x_test[1]), axis=1))])
    # print(distances)

    # x_train = np.broadcast_to(x_train,(len(x_test),)+x_train.shape)
    # x_test = np.expand_dims(x_test, axis=1)
    # print(x_train, np.shape(x_test), '\n\n', x_test, np.shape(x_test))
    # distances = np.sqrt(np.sum(np.square(x_train - x_test), axis=2))
    # print(distances)
    #
    # # iNN = [np.argpartition(distance, range(k))[:k] for distance in distances]
    # iNN = np.argpartition(distances, range(k), axis = 1)
    # print(iNN)
    #
    # y_train = np.broadcast_to(y_train,(len(x_test),)+y_train.shape)
    # print(y_train)
    # yNN = [y_train[i][iNN[i][:k]] for i in range(len(x_test))]
    # print(yNN)
    #
    # kNNValue = [(sum(yNN[i])/len(yNN[i])) for i in range(len(x_test))]
    # print(y_test - kNNValue)

    tree = KDTree(x_train, leaf_size=2)
    dist, ind = tree.query(x_test, k=2)
    print(ind)  # indices of 3 closest neighbors
    print(dist)  # distances to 3 closest neighbors
