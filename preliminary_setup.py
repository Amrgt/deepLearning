#!/usr/bin/env python3

# coding: utf-8

import math
import numpy as np


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


def image2vector(image):
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


def soft_max(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def L1(y_hat, y):
    loss = np.sum(np.abs(y_hat - y))
    return loss


def L2(y_hat, y):
    loss = np.sum(np.dot(y - y_hat, y - y_hat))
    return loss


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    print(np.exp(x))
    print(x + 3)
    sigmoid(x)

    x = np.array([1, 2, 3])
    print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])

    print("image2vector(image) = " + str(image2vector(image)))

    x = np.array([
        [0, 3, 4],
        [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalize_rows(x)))

    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    print("soft_max(x) = " + str(soft_max(x)))

    y_hat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(y_hat, y)))
    print("L2 = " + str(L2(y_hat, y)))
