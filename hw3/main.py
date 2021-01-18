import numpy as np
import random


def load_data(file):
    f = open(file, "r")
    data = f.read()
    f.close()

    all_data = []
    all_label = []

    for line in data.split("\n"):
        sub_data = []
        i = 0
        for value in line.split("\t"):
            if value != "" and i == 10:
                value = float(value)
                all_label.append(value)
                i += 1
            elif value != "":
                value = float(value)
                sub_data.append(value)
                i += 1
        if sub_data != []:
            all_data.append(sub_data)

    for i in range(len(all_data)):
        all_data[i].insert(0, 1)

    all_data = np.array(all_data)
    all_label = np.array(all_label)
    return all_data, all_label


train_data, train_label = load_data("./hw3_train.dat.txt")
test_data, test_label = load_data("./hw3_test.dat.txt")


def linear_regression(x, y):
    x_pinv = np.linalg.pinv(x)
    w = np.dot(x_pinv, y)
    return w


def SGD_LR(x, y, lr, Ein):
    epoch = 0
    Ein_SGD = 100
    w = np.zeros(11)
    while Ein_SGD > Ein*1.01:
        num = random.randint(0, 999)
        w = w + lr * 2 * (y[num] - np.dot(w, x[num].T)) * x[num]
        Ein_SGD = np.square(np.dot(w, x.T) - y)
        Ein_SGD = np.mean(Ein_SGD)
        epoch += 1
    return epoch


# # Q14
# w = linear_regression(train_data, train_label)
# se_arr = []
# for i in range(len(train_data)):
#     pred = np.dot(train_data[i], w)
#     se = (pred - train_label[i])**2
#     se_arr.append(se)
# Ein = np.mean(se_arr)
# print(Ein)


# Q15
# epoch_arr = []
# for i in range(1000):
#     print(i)
#     epoch = SGD_LR(train_data, train_label, 0.001, Ein)
#     epoch_arr.append(epoch)
# ans = np.mean(epoch_arr)
# print(ans)

def theta(s):
    return 1 / (1 + np.exp(-s))


def SGD_Logistic(x, y, lr):
    w = np.zeros(11)
    for i in range(500):
        num = random.randint(0, 999)
        grad = theta(-y[num]*np.dot(w, x[num].T))*(y[num]*x[num])
        w = w + lr*grad
    ce_arr = []
    for j in range(len(x)):
        ce = np.log(1 + np.exp(-y[j]*np.dot(w, x[j].T)))
        ce_arr.append(ce)
    Ein = np.mean(ce_arr)
    return Ein


# Q16
# Ein_arr = []
# for i in range(1000):
#     Ein = SGD_Logistic(train_data, train_label, 0.001)
#     Ein_arr.append(Ein)
# ans = np.mean(Ein_arr)
# print(ans)


def SGD_Logistic_17(x, y, lr):
    w = linear_regression(x, y)
    for i in range(500):
        num = int(np.random.uniform(0, len(x)))
        grad = theta(-y[num]*np.dot(w, x[num].T))*(y[num]*x[num])
        w = w + lr*grad
    ce_arr = []
    for j in range(len(x)):
        ce = np.log(1 + np.exp(-y[j]*np.dot(w, x[j].T)))
        ce_arr.append(ce)
    Ein = np.mean(ce_arr)
    return Ein


# 17
# Ein_arr = []
# for i in range(1000):
#     Ein = SGD_Logistic_17(train_data, train_label, 0.001)
#     Ein_arr.append(Ein)
# ans = np.mean(Ein_arr)
# print(ans)


def sign(number):
    if number > 0:
        return 1
    else:
        return -1


# 18

def Ein_Eout_diff(train_x, train_y, test_x, test_y):
    w = linear_regression(train_x, train_y)
    error = 0
    for i in range(len(train_x)):
        pred = sign(np.dot(w, train_x[i].T))
        if pred != train_y[i]:
            error += 1
    Ein = error / len(train_x)
    error = 0
    for i in range(len(test_x)):
        pred = sign(np.dot(w, test_x[i].T))
        if pred != test_y[i]:
            error += 1
    Eout = error / len(test_x)
    return np.abs(Ein - Eout)


# ans = Ein_Eout_diff(train_data, train_label, test_data, test_label)
# print(ans)

# 19

def trasform(x):
    new_x = []
    for i in range(len(x)):
        tmp = []
        tmp.append(1)
        for j in range(len(x[i])):
            if x[i][j] != 1:
                tmp.append(x[i][j])
                tmp.append(x[i][j]**2)
                tmp.append(x[i][j]**3)
                tmp.append(x[i][j]**4)
                tmp.append(x[i][j]**5)
                tmp.append(x[i][j]**6)
                tmp.append(x[i][j]**7)
                tmp.append(x[i][j]**8)
                tmp.append(x[i][j]**9)
                tmp.append(x[i][j]**10)
        new_x.append(tmp)
    new_x = np.array(new_x)
    return new_x


new_train_data = trasform(train_data)
new_test_data = trasform(test_data)

ans = Ein_Eout_diff(new_train_data, train_label, new_test_data, test_label)
print(ans)
