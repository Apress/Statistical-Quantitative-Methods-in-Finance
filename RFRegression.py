import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import statsmodels.api as sm

logging.basicConfig(level=logging.DEBUG)


class RandomForestPredictor:
    PERIOD = 1
    PRICE_COL = "Close"
    VOLUME_COL = "Volume"

    def __init__(self, dirname, security, trainTestRatio=0.9, maxTrees=200, batchSize=32):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dirname = dirname
        self.security = security
        self.maxTrees = maxTrees
        self.batchSize = batchSize
        self.df = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        self.endog, self.exog = None, None
        self.beginIndex = None
        self.endIndex = None
        self.calculateEndogExogVars()
        self.ntraining = int(trainTestRatio * self.df.shape[0])
        self.nn = None
        self.ols = self.createOLSModel()
        self.rf = None

    def movingAverage(self, arr, period):
        result = np.zeros(len(arr), dtype=np.float32)
        sum1 = np.sum(arr[0:period])
        for i in range(period, len(arr), 1):
            result[i] = sum1 / period
            sum1 += arr[i] - arr[i-period]
        return result

    def volatility(self, arr, lookback):
        result = np.zeros(len(arr), dtype=np.float32)
        sumsq = np.sum(arr[0:lookback] ** 2)
        for i in range(lookback, len(arr), 1):
            result[i] = sumsq / lookback
            sumsq += arr[i]*arr[i] - arr[i-lookback]*arr[i-lookback]
        return result

    def calculateEndogExogVars(self):
        prices = self.df.loc[:, self.PRICE_COL].values
        returns = prices[self.PERIOD:] / prices[0:-self.PERIOD] - 1
        self.df.loc[:, "returns"] = 0
        self.df.loc[0:self.df.shape[0] - 1 - self.PERIOD, "returns"] = returns
        self.endog = "returns"

        self.df.loc[:, "lag1Return"] = 0
        self.df.loc[self.PERIOD+1:, "lag1Return"] = returns[0:self.df.shape[0]-self.PERIOD-1]

        self.df.loc[:, "lag2Return"] = 0
        self.df.loc[self.PERIOD+2:, "lag2Return"] = returns[0:self.df.shape[0]-self.PERIOD-2]

        self.df.loc[:, "lag3Return"] = 0
        self.df.loc[self.PERIOD+3:, "lag3Return"] = returns[0:self.df.shape[0]-self.PERIOD-3]

        self.df.loc[:, "ma3m5"] = 0
        ma3 = self.movingAverage(prices, 3)
        ma5 = self.movingAverage(prices, 5)
        self.df.loc[5:, "ma3m5"] = ma3[5:] - ma5[5:]

        volatility = self.volatility(returns, lookback=5)
        moVolatility = self.volatility(returns, lookback=21)
        relVolat = volatility[21:] / moVolatility[21:]
        self.df.loc[:, "relVolatility"] = 0
        self.df.loc[21:self.df.shape[0] - 1 - self.PERIOD, "relVolatility"] = relVolat

        volume = self.df.loc[:, self.VOLUME_COL].values
        vol3 = self.movingAverage(volume, 3)
        vol5 = self.movingAverage(volume, 5)
        relVolume = vol3[5:] / vol5[5:]
        self.df.loc[:, "relVolume"] = 0
        self.df.loc[5:, "relVolume"] = relVolume

        self.exog = ["lag1Return", "lag2Return", "lag3Return", "ma3m5", "relVolatility", "relVolume"]
        self.beginIndex = 21
        self.endIndex = self.df.shape[0] - self.PERIOD

    def fitRF(self, ntrees):
        self.rf = self.rf = RandomForestRegressor(n_estimators=ntrees, random_state=0)
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        self.rf = self.rf.fit(X, y)
        yhat = self.rf.predict(X)
        rmseRF = np.sqrt(np.mean((y - yhat) ** 2))
        return rmseRF

    def createOLSModel(self):
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        return sm.OLS(endog=y, exog=X)

    def fitOLS(self):
        self.ols = self.ols.fit()
        return self.ols

    def testRF(self, y, X):
        yhatRF = self.rf.predict(X)
        rmseRF = np.sqrt(np.mean((y - yhatRF) ** 2))
        return rmseRF

    def testOLS(self, y, X):
        Xols = sm.add_constant(X, has_constant="add")
        yhatOls = self.ols.predict(exog=Xols)
        rmseOLS = np.sqrt(np.mean((y - yhatOls) ** 2))
        return rmseOLS

    def plot(self, trees, trainError, testError, olsErrorTrain, olsErrorTest):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(trees, trainError, label="RF Training RMSE")
        axs.plot(trees, testError, label="RF Testing RMSE")
        axs.axhline(y=olsErrorTrain, color='r', linestyle='dashed', label="OLS Training RMSE")
        axs.axhline(y=olsErrorTest, color='g', linestyle='dashdot', label="OLS Testing RMSE")
        axs.set(title="Selecting Number of Estimators (Trees) for Random Forest")
        axs.legend()
        axs.grid()
        axs.set_xlabel("Estimators")
        axs.set_ylabel("RMSE")
        plt.savefig(os.path.join(self.dirname, f"AssetReturnRF_{self.security}.jpeg"),
                    dpi=500)
        plt.show()

    def findOptimalTrainingEstimators(self):
        ntrees = list(range(10, self.maxTrees, 10))
        testError = []
        trainError = []
        self.fitOLS()
        ytrain = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        Xtrain = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        ytest = self.df.loc[self.ntraining:self.endIndex - 1, self.endog].values
        Xtest = self.df.loc[self.ntraining:self.endIndex - 1, self.exog].values
        olsErrorTrain = self.testOLS(ytrain, Xtrain)
        olsErrorTest = self.testOLS(ytest, Xtest)
        for ntree in ntrees:
            nnerror = self.fitRF(ntrees=ntree)
            self.logger.info("Estimators: %d, Fitting RMSE: %f", ntree, nnerror)
            rfErrorTrain = self.testRF(ytrain, Xtrain)
            rfErrorTest = self.testRF(ytest, Xtest)
            testError.append(rfErrorTest)
            trainError.append(rfErrorTrain)
        self.plot(ntrees, trainError, testError, olsErrorTrain, olsErrorTest)
        self.logger.info("OLS RMS error on training dataset: %f, testing dataset: %f", olsErrorTrain, olsErrorTest)


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    pred = RandomForestPredictor(dirname, "SPY")
    np.random.seed(32)
    pred.findOptimalTrainingEstimators()