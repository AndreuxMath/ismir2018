import torch
import numpy as np


def mmd_linear(X, Y):
    """
    X: shape n * p
    Y: shape m * p
    returns energy distance for linear kernel k(x, y) = x^T y,
    such that
    MMD_k(X, Y) = 1/n(n - 1) sum_{i \neq j} k(x_i, x_j)
                + 1/m(m - 1) sum_{i \neq j} k(y_i, y_j)
                - 2 / (m * n) \sum_{i, j} k(x_i, y_j)
    """
    n = X.shape[0]
    gram_X = np.dot(X, X.T)
    gram_X[np.arange(n), np.arange(n)] = 0.
    m = Y.shape[0]
    gram_Y = np.dot(Y, Y.T)
    gram_Y[np.arange(m), np.arange(m)] = 0.

    cross_XY = np.dot(X, Y.T)

    term1 = np.sum(gram_X) / float(n * (n - 1))
    term2 = np.sum(gram_Y) / float(m * (m - 1))
    term3 = np.sum(cross_XY) / float(m * n)

    return term1 + term2 - 2 * term3


def mmd_linear_th(X, Y):
    """
    X: size n * p
    Y: size m * p
    returns energy distance for linear kernel k(x, y) = x^T y,
    such that
    MMD_k(X, Y) = 1/n(n - 1) sum_{i \neq j} k(x_i, x_j)
                + 1/m(m - 1) sum_{i \neq j} k(y_i, y_j)
                - 2 / (m * n) \sum_{i, j} k(x_i, y_j)
    """
    n = X.size(0)
    gram_X = torch.matmul(X, X.transpose(0, 1))
    gram_X = gram_X - torch.diag(torch.diag(gram_X))

    m = Y.size(0)
    gram_Y = torch.matmul(Y, Y.transpose(0, 1))
    gram_Y = gram_Y - torch.diag(torch.diag(gram_Y))

    cross_XY = torch.matmul(X, Y.transpose(0, 1))

    term1 = torch.sum(gram_X) / float(n * (n - 1))
    term2 = torch.sum(gram_Y) / float(m * (m - 1))
    term3 = torch.sum(cross_XY) / float(m * n)

    return term1 + term2 - 2 * term3
