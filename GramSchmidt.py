import numpy as np


class GramSchmidt(object):
    def __init__(self, dtype=np.int32):
        self.dtype = dtype

    def qrDecomSimple(self, matrix):
        nrow, ncol = matrix.shape
        Q1 = matrix.copy()
        R = np.zeros((ncol, ncol), dtype=self.dtype)
        for j in range(ncol):
            for i in range(j):
                R[i, j] = np.dot(Q1[:, i], matrix[:, j])
                Q1[:, j] -= R[i, j] * Q1[:, i]
            R[j, j] = np.dot(Q1[:, j], Q1[:, j])
            Q1[:, j] /= R[j, j]
        return Q1, R

    def qrDecompose(self, matrix):
        ''' Modified (stable) version of Gram Schmidt decomposition '''
        nrow, ncol = matrix.shape
        Q1 = matrix.copy()
        R = np.zeros((ncol, ncol), dtype=self.dtype)
        for j in range(ncol):
            for i in range(j):
                R[i, j] = np.dot(Q1[:, i], Q1[:, j])
                Q1[:, j] -= R[i, j] * Q1[:, i]
            R[j, j] = np.dot(Q1[:, j], Q1[:, j])
            Q1[:, j] /= R[j, j]
        return Q1, R

