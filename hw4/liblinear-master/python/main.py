import numpy as np
from liblinearutil import *


def split_data_label(data):
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


def split_train_val(data, train_percentage):
    num = int(len(data)*train_percentage)
    train_data = data[:num]
    val_data = data[num:]
    return train_data, val_data


def transform_data(data, d):
    new_data = []
    for i in range(len(data)):
        tmp = []
        tmp.append(1)
        for j in range(len(data[i])):
            tmp.append(data[i][j])
        for j in range(len(data[i])):
            num = d - j
            for m in range(num):
                xx = data[i][j]*data[i][j+m]
                tmp.append(xx)
        new_data.append(tmp)
    new_data = np.array(new_data)
    return new_data


all_train_data = np.loadtxt("./hw4_train.dat.txt")
train_data, train_label = split_data_label(all_train_data)
train_data = transform_data(train_data, 6)

all_test_data = np.loadtxt("./hw4_test.dat.txt")
test_data, test_label = split_data_label(all_test_data)
test_data = transform_data(test_data, 6)

# 16
lambdas = [0.0001, 0.01, 1, 100, 10000]
best_acc = 0.0
best_lambda = 0
for lambda_ in lambdas:
    C = 1 / (2*lambda_)
    m = train(train_label, train_data, "-s 0 -c %f -e 0.000001" % C)
    p_label, p_acc, p_val = predict(test_label, test_data, m)
    ACC, MSE, SCC = evaluations(test_label, p_label)
    if ACC >= best_acc:
        best_acc = ACC
        best_lambda = 1 / (2*C)

print("best_lambda:", best_lambda)

# 17
lambdas = [0.0001, 0.01, 1, 100, 10000]
best_acc = 0.0
best_lambda = 0
for lambda_ in lambdas:
    C = 1 / (2*lambda_)
    m = train(train_label, train_data, "-s 0 -c %f -e 0.000001" % C)
    p_label, p_acc, p_val = predict(train_label, train_data, m)
    ACC, MSE, SCC = evaluations(train_label, p_label)
    if ACC >= best_acc:
        best_acc = ACC
        best_lambda = 1 / (2*C)

print("best_lambda:", best_lambda)

# 18, 19
train_data, val_data = split_train_val(train_data, 0.6)
train_label, val_label = split_train_val(train_label, 0.6)
lambdas = [0.0001, 0.01, 1, 100, 10000]
best_acc = 0.0
best_lambda = 0
for lambda_ in lambdas:
    C = 1 / (2*lambda_)
    m = train(train_label, train_data, "-s 0 -c %f -e 0.000001" % C)
    p_label, p_acc, p_val = predict(val_label, val_data, m)
    ACC, MSE, SCC = evaluations(val_label, p_label)
    if ACC >= best_acc:
        best_acc = ACC
        best_lambda = 1 / (2*C)

C = 1 / (2*best_lambda)
m = train(train_label, train_data, "-s 0 -c %f -e 0.000001" % C)
p_label, p_acc, p_val = predict(test_label, test_data, m)
ACC, MSE, SCC = evaluations(test_label, p_label)
print("18 best_E:", 100 - ACC)

train_data = np.concatenate([train_data, val_data])
train_label = np.concatenate([train_label, val_label])
C = 1 / (2*best_lambda)
m = train(train_label, train_data, "-s 0 -c %f -e 0.000001" % C)
p_label, p_acc, p_val = predict(test_label, test_data, m)
ACC, MSE, SCC = evaluations(test_label, p_label)

print("19 best_E:", 100 - ACC)


# 20
num = int(len(all_train_data) / 5)
all_acc = []
for i in range(5):
    val_data = train_data[num*i:num*(i+1)]
    val_label = train_label[num*i:num*(i+1)]
    nums = np.arange(num*i, num*(i+1))
    data = train_data
    label = train_label
    data = np.delete(data, nums, 0)
    label = np.delete(label, nums, 0)
    lambdas = [0.0001, 0.01, 1, 100, 10000]
    best_acc = 0.0
    best_lambda = 0
    for lambda_ in lambdas:
        C = 1 / (2*lambda_)
        m = train(label, data, "-s 0 -c %f -e 0.000001" % C)
        p_label, p_acc, p_val = predict(val_label, val_data, m)
        ACC, MSE, SCC = evaluations(val_label, p_label)
        if ACC >= best_acc:
            best_acc = ACC
            best_lambda = 1 / (2*C)
    all_acc.append(best_acc)

best_E = (100 - np.mean(all_acc)) / 100
print(best_E)
print(all_acc)
