# import pandas
from data_utils import load_dataset


if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # print(str(x_train[2][0])+', '+str(y_train[2][0]))
    xy_train_list = ['x: ' + str(x_train[i][0]) + ', y: ' + str(y_train[i][0]) for i in range(0, len(x_train))]
    print("\n".join(xy_train_list))
