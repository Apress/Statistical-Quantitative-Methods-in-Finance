import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import scipy.stats as ss

logging.basicConfig(level=logging.DEBUG)


class BinomialModel(ABC):
    PRICE_COL = "Close"
    PERIOD = 5

    def __init__(self, dirname, security, trainTestSplit=0.9, seed=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        df = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        df = self.calculateReturns(df)
        self.df = df
        self.dirname = dirname
        self.security = security
        self.trainingRows = int(df.shape[0] * trainTestSplit)
        np.random.seed(seed)

    def calculateReturns(self, df):
        price = df.loc[:, self.PRICE_COL].values
        returnCol = price[self.PERIOD:] / price[:-self.PERIOD] - 1
        df.loc[:, "return"] = 0
        df.loc[0:df.shape[0] - self.PERIOD - 1, "return"] = returnCol
        df = df.loc[0:df.shape[0] - self.PERIOD - 1, :].reset_index(drop=True)
        return df

    @abstractmethod
    def fit(self, endIndex=None):
        raise NotImplementedError(f"Sub class {self.__class__.__name__} needs to implement")

    @abstractmethod
    def predict(self, index):
        raise NotImplementedError(f"Sub class {self.__class__.__name__} needs to implement")

    @abstractmethod
    def testHypothesis(self, theta=None, nobservation=None, nsuccess=None):
        raise NotImplementedError(f"Sub class {self.__class__.__name__} needs to implement")

    def test(self):
        actual = np.zeros(self.df.shape[0] - self.trainingRows, dtype=np.int8)
        predicted = np.zeros(actual.shape[0], dtype=np.int8)
        thetaArr = np.zeros(actual.shape[0], dtype=np.float32)
        for i in range(self.trainingRows, self.df.shape[0], 1):
            predicted[i - self.trainingRows] = self.predict(i)
            actual[i - self.trainingRows] = np.where(self.df.loc[i, "return"] > 0, 1, 0)
            thetaArr[i - self.trainingRows] = self.theta
            self.fit(i)

        accuracy = (sum(actual == (predicted)) / actual.shape[0]) * 100
        self.logger.info("Overall accuracy of %s: %.2f", self.__class__.__name__, accuracy)
        return thetaArr

    def compareResults(self, freq, bayes):
        dates = self.df.loc[self.trainingRows:, "Date"].values
        fig, ax = plt.subplots(nrows=2, figsize=(10, 7))
        ax[0].plot(dates, freq, label="Frequentist")
        ax[0].plot(dates, bayes, label="Bayesian")
        ax[0].set(title="Frequentist and Bayesian Values of Parameter Theta")
        ax[0].set_ylabel("Theta")
        ax[0].set_xlabel("Date")
        ax[0].legend()
        ax[0].grid()

        diffs = freq - bayes
        ax[1].hist(diffs, bins=20)
        ax[1].set(title="Histogram of Difference Between Frequentist and Bayesian Predictions")
        ax[1].grid()
        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"diff_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()


class Frequentist(BinomialModel):
    def fit(self, endIndex=None):
        if endIndex is None:
            endIndex = self.trainingRows
        returns = self.df.loc[0:endIndex-1, "return"].values
        self.theta = np.sum(returns > 0) / returns.shape[0]
        self.logger.info("class: %s, Date: %s, theta = %f", self.__class__.__name__, str(self.df.loc[endIndex, "Date"]), self.theta)

    def predict(self, index):
        return np.random.binomial(1, self.theta, 1)

    def testHypothesis(self, theta=None, nobservation=None, nsuccess=None):
        if theta is None:
            theta = self.theta
        if nobservation is None:
            nobservation = self.df.shape[0]
        if nsuccess is None:
            nsuccess = sum(self.df.loc[:, "return"].values > 0)
        result = ss.binom_test(nsuccess, nobservation, theta)
        self.logger.info("P-value: %f", result)


class Bayesian(BinomialModel):
    def __init__(self, dirname, security, trainTestSplit=0.9, seed=10, alpha=0.5, beta=0.5):
        super().__init__(dirname, security, trainTestSplit, seed)
        self.alpha = alpha
        self.beta = beta

    def fit(self, endIndex=None):
        if endIndex is None:
            endIndex = self.trainingRows
        returns = self.df.loc[0:endIndex-1, "return"].values
        N = returns.shape[0]
        k = np.sum(returns > 0)
        self.theta = (k + self.alpha - 1) / (N + self.alpha + self.beta - 2)
        self.logger.info("class: %s, Date: %s, theta = %f", self.__class__.__name__, str(self.df.loc[endIndex, "Date"]), self.theta)

    def predict(self, index):
        return np.random.binomial(1, self.theta, 1)

    def testHypothesis(self, theta=None, nobservation=None, nsuccess=None):
        if theta is None:
            theta = self.theta
        if nobservation is None:
            nobservation = self.df.shape[0]
        if nsuccess is None:
            nsuccess = sum(self.df.loc[:, "return"].values > 0)
        result = ss.binom_test(nsuccess, nobservation, theta)
        self.logger.info("P-value: %f", result.pvalue)


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    freqModel = Frequentist(dirname, "BAC")
    freqModel.fit()
    freq = freqModel.test()
    freqModel.testHypothesis()

    bayesianModel = Bayesian(dirname, "BAC", alpha=0.5, beta=0.5)
    bayesianModel.fit()
    bayes = bayesianModel.test()

    freqModel.compareResults(freq, bayes)