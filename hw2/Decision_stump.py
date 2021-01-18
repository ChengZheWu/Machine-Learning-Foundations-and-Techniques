import random
import numpy as np
from tqdm import tqdm


def sign(number):
    if number > 0:
        return 1
    else:
        return -1


def get_data(N, probability):
    x = np.random.uniform(-1, 1, (N, 1))
    y = np.zeros((N, 1))
    arr = np.random.permutation([i for i in range(N)])

    for i in range(N):
        if i < N*probability:
            y[arr[i]][0] = -sign(x[arr[i]][0])
        else:
            y[arr[i]][0] = sign(x[arr[i]][0])
    data = np.concatenate((x, y), axis=1)
    data = np.sort(data, axis=0)

    return data


def decision_stump(data):
    N = len(data)
    min_errors = N
    min_s = 0
    min_theta = 0

    for s in [-1, 1]:
        for i in range(N):
            if i >= 0 and i <= N-2:
                if data[i][0] != data[i+1][0]:
                    theta = (data[i][0] + data[i+1][0]) / 2
                else:
                    theta = -1
            else:
                theta = -1

            errors = 0
            for j in range(N):
                pred = s*sign(data[j][0] - theta)
                if pred != data[j][1]:
                    errors += 1
            if errors < min_errors:
                min_errors = errors
                min_s = s
                min_theta = theta
            if errors == min_errors:
                if s + theta < min_s + min_theta:
                    min_s = s
                    min_theta = theta

    error_rate = min_errors / N

    return error_rate, min_s, min_theta


def CalEout(data, s, theta):
    N = len(data)
    errors = 0
    for i in range(N):
        pred = s*sign(data[i][0] - theta)
        if pred != data[i][1]:
            errors += 1

    error_rate = errors / N

    return error_rate


def train(N, probability):
    E_list = []
    for i in tqdm(range(100), ncols=50):

        # Ein
        N = 20
        data = get_data(N, probability)
        E_in, s, theta = decision_stump(data)

        # Eout
        N = 100000
        data = get_data(N, probability)
        E_out = CalEout(data, s, theta)

        E_diff = E_out - E_in
        E_list.append(E_diff)

    E_mean = np.mean(E_list)

    print(E_mean)


train(2, 0)
train(20, 0)
train(2, 0.1)
train(20, 0.1)
train(200, 0.1)
