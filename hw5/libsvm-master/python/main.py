import numpy as np
import random
from svmutil import *
from svm import *


def read_data(file):
    f = open(file)
    data = f.read()
    f.close()

    all_data = []
    all_label = []
    for line in data.split("\n"):
        i = 0
        line_data = []
        for sub_line in line.split(" "):
            for value in sub_line.split(":"):
                if i == 0:
                    if value != "":
                        all_label.append(int(value))
                elif i != 0 and i % 2 == 0:
                    if value != "":
                        line_data.append(float(value))
                i += 1
        if line_data != []:
            all_data.append(line_data)
    return all_data, all_label


def redefine_label(label, class_number):
    new_label = []
    for i in range(len(label)):
        if label[i] == class_number:
            new_label.append(1)
        else:
            new_label.append(-1)
    return new_label


train_data, train_label = read_data("./train.txt")
test_data, test_label = read_data("./test.txt")
# print(len(train_data), len(train_label))
# print(len(test_data), len(test_label))


# 15
train_label = redefine_label(train_label, 3)
test_label = redefine_label(test_label, 3)
m = svm_train(train_label, train_data, "-c 10 -t 0")
print(m)

# 16, 17
# train_label = redefine_label(train_label, 2)
# test_label = redefine_label(test_label, 2)
# m = svm_train(train_label, train_data, "-c 10 -t 1 -d 2")
# p_label, p_acc, p_val = svm_predict(train_label, train_data, m)

# 18
# best_Eout = 100
# best_c = 0
# Cs = [0.01, 0.1, 1, 10, 100]
# train_label = redefine_label(train_label, 6)
# test_label = redefine_label(test_label, 6)
# for c in Cs:
#     m = svm_train(train_label, train_data, "-c %f -t 2 -g 10" % c)
#     p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
#     print(p_acc[0])
#     if 100 - p_acc[0] < best_Eout:
#         best_Eout = 100 - p_acc[0]
#         best_c = c
# print("best:", best_Eout, best_c)

# 19
# best_Eout = 100
# best_gamma = 10000
# gammas = [0.1, 1, 10, 100, 1000]
# train_label = redefine_label(train_label, 6)
# test_label = redefine_label(test_label, 6)
# for gamma in gammas:
#     m = svm_train(train_label, train_data, "-c 0.1 -t 2 -g %f" % gamma)
#     p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
#     print(p_acc[0])
#     if 100 - p_acc[0] < best_Eout:
#         best_Eout = 100 - p_acc[0]
#         best_gamma = gamma
# print("best:", best_Eout, best_gamma)

# 20
# def train_val_split(data, label, val_index):
#     train_data = []
#     train_label = []
#     val_data = []
#     val_label = []
#     for i in range(len(data)):
#         if i in val_index:
#             val_data.append(data[i])
#             val_label.append(label[i])
#         else:
#             train_data.append(data[i])
#             train_label.append(label[i])
#     return train_data, train_label, val_data, val_label


# train_label = redefine_label(train_label, 6)
# test_label = redefine_label(test_label, 6)
# all_index = np.arange(0, len(train_data))
# val_index = random.sample(range(0, len(train_data)), 200)
# train_data, train_label, val_data, val_label = train_val_split(
#     train_data, train_label, val_index)

# gammas = [0.1, 1, 10, 100, 1000]
# smallest_gamma = []

# best_Eout = 100
# best_gamma = 10000
# for epoch in range(10):
#     print(epoch)
#     for gamma in gammas:
#         m = svm_train(train_label, train_data, "-c 0.1 -t 2 -g %f" % gamma)
#         p_label, p_acc, p_val = svm_predict(val_label, val_data, m)
#         if 100 - p_acc[0] <= best_Eout:
#             if gamma < best_gamma:
#                 best_gamma = gamma
#     smallest_gamma.append(best_gamma)
# print(smallest_gamma)
# counts = np.bincount(smallest_gamma)
# print(np.argmax(counts))
