import numpy as np


def sign(number):
    if number > 0:
        return 1
    else:
        return -1


def unit(vector):
    norm2 = np.linalg.norm(vector, keepdims=True)
    unit_vector = vector / norm2
    return unit_vector


def pla(x, y, w):  # Percentron Learnong Algorithm
    epoch = 0
    train = True
    while train:
        pred = []
        epoch += 1
        index_list = np.arange(0, 100)
        np.random.shuffle(index_list)  # 打亂每次取資料的順序
        for i in index_list:
            y_pred = sign(np.dot(w.T, x[i]))
            if y_pred != y[i]:  # 找到錯誤就更新
                w = w + y[i]*x[i]
                break
        for i in range(len(x)):
            pred.append(sign(np.dot(w.T, x[i])))
        pred = np.array(pred)

        if (y == pred).all():
            train = False
            return epoch, w
        else:
            continue


# 讀取檔案
f = open(".\hw1_train.dat.txt", "r")
data = f.read()
f.close()

all_data = []

for line in data.split("\n"):
    sub_data = []
    for value in line.split("\t"):
        if value != "":
            value = float(value)
            sub_data.append(value)
    if sub_data != []:
        all_data.append(sub_data)

x = []
y = []
for i in range(len(all_data)):
    tmp = []
    for j in range(10):
        tmp.append(all_data[i][j])
    x.append(tmp)
    y.append(all_data[i][10])

# 設置x0
for i in range(len(x)):
    x[i].insert(0, 1)  # set x0 = 1  # 第16, 17題
    # x[i].insert(0, 10)  # 第18題, set x0 = 10
    # x[i].insert(0, 0)  #第19題 set x0 = 0


# 第20題
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         x[i][j] = x[i][j]/4

x = np.array(x)
y = np.array(y, dtype=np.int8)

# 開始trian
w0_list = []  # 存放所有的W0
epoch_list = []  # 存放所有的訓練圈數
for num in range(1000):
    w = np.zeros(11)  # 初始權重
    epoch, w = pla(x, y, w)
    w0_list.append(w[0])
    epoch_list.append(epoch)

print(np.median(epoch_list))
print(np.median(w0_list))
