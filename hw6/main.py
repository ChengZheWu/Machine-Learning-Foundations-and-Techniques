import numpy as np
import pandas as pd
import random
from tqdm import tqdm


def split_data_label(data):
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


data = np.loadtxt("./hw6_train.dat.txt")
x, y = split_data_label(data)
print(x.shape, y.shape)

data = np.loadtxt("./hw6_test.dat.txt")
x_test, y_test = split_data_label(data)
print(x_test.shape, y_test.shape)


class CART:
    def __init__(self):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0
        self.root = None

    def fit(self, features, target):
        self.root = CART()
        self.root._grow_tree(features, target)

    def predict(self, features):
        return np.array([self.root._predict(f) for f in features])

    def _grow_tree(self, features, target):
        self.n_samples = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        self.label = max([(c, len(target[target == c]))
                          for c in np.unique(target)], key=lambda x: x[1])[0]

        impurity_node = self._calc_impurity(target)

        for col in range(features.shape[1]):
            feature_level = np.unique(features[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[features[:, col] <= threshold]
                impurity_l = self._calc_impurity(target_l)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[features[:, col] > threshold]
                impurity_r = self._calc_impurity(target_r)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - \
                    (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self._split_tree(features, target)

    def _split_tree(self, features, target):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._grow_tree(features_l, target_l)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._grow_tree(features_r, target_r)

    def _calc_impurity(self, target):
        return 1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])

    def _predict(self, d):
        if self.feature != None:
            if d[self.feature] <= self.threshold:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else:
            return self.label


# 14
cart = CART()
cart.fit(x, y)
preds = cart.predict(x_test)
E = 1 - sum(preds == y_test)/len(y_test)
print(E)


def boostrap(x, y, N):
    indexs = [random.randint(0, N//2) for _ in range(N)]
    return x[indexs], y[indexs]


def predict(x, y, x_test, T):
    pred_tmp = np.zeros(y.shape)
    final_pred = []
    for i in tqdm(range(T)):
        tx, ty = boostrap(x, y, len(y)-1)
        cart = CART()
        cart.fit(tx, ty)
        pred = cart.predict(x_test)
        pred_tmp += pred
    for i in range(len(pred_tmp)):
        if pred_tmp[i] >= 0:
            final_pred.append(1)
        else:
            final_pred.append(-1)
    return np.array(final_pred)


# 15
pred = predict(x, y, x_test, 10)
E = 1 - sum(pred == y_test)/len(y_test)
print(E)


def boostrap(x, y, N):
    indexs = [random.randint(0, N//2) for _ in range(N)]
    return x[indexs], y[indexs]


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def predict(x, y, x_test, T):
    pred_tmp = np.zeros(y.shape)
    final_pred = []
    for i in tqdm(range(T)):
        tx, ty = boostrap(x, y, len(y)-1)
        cart = CART()
        cart.fit(tx, ty)
        pred = cart.predict(x_test)
        pred_tmp += pred
    for i in range(len(pred_tmp)):
        final_pred.append(sign(pred_tmp[i]))
    return np.array(final_pred)


# 16
pred = predict(x, y, x, 10)
E = 1 - sum(pred == y)/len(y)
print(E)

# 17
pred = predict(x, y, x_test, 10)
E = 1 - sum(pred == y_test)/len(y_test)
print(E)
