import numpy as np


class HouseHolder(object):
    def __init__(self, dtype=np.int32):
        self.dtype = dtype

    def houseHolderVecToQ(self, V):
        X0 = np.eye(V.shape[0], dtype=self.dtype)
        factor = -2 / np.dot(V[:, 0], V[:, 0])
        X0 += factor * np.einsum("i,j->ij", V[:, 0], V[:, 0])
        for j in range(1, V.shape[1]):
            factor = -2 / np.dot(V[:, j], V[:, j])
            X1 = np.eye(V.shape[0], dtype=self.dtype)
            X1 += factor * np.einsum("i,j->ij", V[:, 0], V[:, 0])
            X0 = np.einsum("ij,jk->ik", X1, X0)
        return X0

    def qrDecompose(self, matrix):
        nrow, ncol = matrix.shape
        V = matrix.copy()
        R = np.zeros((ncol, ncol), dtype=self.dtype)
        for j in range(ncol):
            R[0:j+1, j] = V[0:j+1, j]
            V[0:j, j] = 0
            v = V[:, j]
            xlen = np.dot(v, v)
            mult = 1 if v[j] >= 0 else -1
            v[j] += mult * xlen
            # v is now the vector in householder transformation
            vlen = np.dot(v, v)
            for i in range(j+1, ncol):
                V[:, i] += - 2 * (np.dot(v, V[:, i]) / vlen) * v

        Q = self.houseHolderVecToQ(V)
        return Q, R


