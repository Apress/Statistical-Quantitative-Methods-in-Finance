import numpy as np
import statsmodels.api as sm
import logging
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import os


logging.basicConfig(level=logging.DEBUG)


class CensoredData(object):
    def __init__(self, dirname, coeff=3, variance=1, l1=0.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        x = np.arange(-5, 5, 0.01)
        sdev = np.sqrt(variance)
        epsilon = sdev * np.random.standard_normal(x.shape[0])
        ystar = 1 + coeff*x + epsilon
        y = np.where(ystar < l1, l1, ystar)
        self.x = x
        self.y = y
        self.dirname = dirname

    def fitOLS(self):
        x = sm.add_constant(self.x[:, np.newaxis], has_constant="add")
        olsModel = sm.OLS(self.y, x).fit()
        self.logger.info(olsModel.summary())
        predicted = olsModel.predict(x)

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        colors = cm.rainbow(np.linspace(0, 1, 2))
        axs.scatter(self.x, self.y, c=colors[0], label="Observed")
        axs.scatter(self.x, predicted, c=colors[1], label="OLS Predicted")
        axs.grid()
        axs.legend()
        axs.set_xlabel("X")
        axs.set_ylabel("y")
        axs.set(title="Spurious Fitting of Censored Data Using OLS")
        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"plot_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    censoredData = CensoredData(dirname)
    censoredData.fitOLS()
