import numpy as np


def split_data_label(data):
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


def transform_data(data, d):
    new_data = []
    for i in range(len(data)):
        tmp = []
        tmp.append(1)
        for j in range(len(data[i])):
            tmp.append(data[i][j])
            num = d - j
            for m in range(num):
                xx = data[i][j]*data[i][j+m]
                tmp.append(xx)
        new_data.append(tmp)
    new_data = np.array(new_data)
    return new_data


data = np.loadtxt("./hw4_train.dat.txt")
data, label = split_data_label(data)
print(data.shape)
print(label.shape)

new_data = transform_data(data, 6)
print(new_data.shape)
