import numpy as np
from scipy import integrate


def f(x):
    return np.exp(x)


def b(x):
    return ((3 + np.exp(2)) / 8)*x


def c(x):
    return ((3 + 3*np.exp(2)) / 8)*x


def d(x):
    return (np.exp(2) / 8)*x


def e(x):
    return (3*np.exp(2) / 8)*x


area_f, _ = integrate.quad(f, 0, 2)
area_b, _ = integrate.quad(b, 0, 2)
area_c, _ = integrate.quad(c, 0, 2)
area_d, _ = integrate.quad(d, 0, 2)
area_e, _ = integrate.quad(e, 0, 2)

print(area_f - area_b)
print(area_f - area_c)
print(area_f - area_d)
print(area_f - area_e)
