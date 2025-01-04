import numpy as np


def relu(x: np.ndarray):
    return np.where(x > 0, x, np.zeros_like(x))


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


x = np.array([[-1, -2, 3], [4, 5, -6]])
# [[0 0 3]
# [4 5 0]]
print(relu(x))

y = np.array([3.0, 1.0, 0.2])
print(softmax(y))
