import numpy as np


class Choice(object):
    def sim(self, N):
        arr = np.arange(N)
        res = np.random.choice(arr, N, replace=True)
        return np.unique(res).shape[0] / float(N)


if __name__ == "__main__":
    np.random.seed(32)
    ch = Choice()
    res = [ch.sim(k) for k in range(2000, 2005)]
    print(res)