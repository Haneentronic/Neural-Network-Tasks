import numpy as np
import random


def perceptron(x, t, ephocs, learningrate, b):
    if b == 1:
        b = random.random()
    w = np.random.randn(2)
    for n in range(ephocs):
        for i in range(x.shape[0]):
            y = np.sign(np.dot(x.iloc[i, :], w.T) + b)
            if y != t[i]:
                loss = t[i] - y
                w = w + (np.dot(loss * learningrate, x.iloc[i, :]))
    return w
