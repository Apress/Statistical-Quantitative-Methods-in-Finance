import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import statsmodels.api as sm

logging.basicConfig(level=logging.DEBUG)


class AssetReturnPredictor:
    PERIOD = 1
    PRICE_COL = "Close"
    VOLUME_COL = "Volume"

    def __init__(self, dirname, security, trainTestRatio=0.9, maxEpochs=100, batchSize=32):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dirname = dirname
        self.security = security
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.df = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        self.endog, self.exog = None, None
        self.beginIndex = None
        self.endIndex = None
        self.calculateEndogExogVars()
        self.ntraining = int(trainTestRatio * self.df.shape[0])
        self.nn = None
        self.ols = self.createOLSModel()
        self.olsFitted = False

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

    def createNN(self):
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.BatchNormalization())
        nn.add(tf.keras.layers.Dense(30, activation=tf.keras.activations.tanh))
        nn.add(tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu))
        nn.add(tf.keras.layers.Dense(30, activation=tf.nn.leaky_relu))
        nn.add(tf.keras.layers.Dense(20))
        nn.add(tf.keras.layers.Dense(5))
        nn.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh))
        nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError())
        return nn

    def fitNN(self, nepochs):
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        Xy = np.concatenate((X, y[:, np.newaxis]), axis=1)
        np.random.shuffle(Xy)
        X = Xy[:, 0:-1]
        y = Xy[:, -1]
        self.nn = self.createNN()
        return self.nn.fit(X, y, batch_size=self.batchSize, epochs=nepochs)

    def createOLSModel(self):
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        return sm.OLS(endog=y, exog=X)

    def fitOLS(self):
        if self.olsFitted:
            return self.ols
        self.ols = self.ols.fit()
        self.olsFitted = True
        return self.ols

    def fit(self, nepochs):
        self.fitOLS()
        nnFitHistory = self.fitNN(nepochs)
        return np.sqrt(nnFitHistory.history["loss"][-1])

    def testNN(self, y=None, X=None):
        if y is None:
            y = self.df.loc[self.ntraining:self.endIndex-1, self.endog].values
            X = self.df.loc[self.ntraining:self.endIndex-1, self.exog].values
        yhatNN = self.nn.predict(X)
        rmseNN = np.sqrt(np.mean((y - yhatNN) ** 2))
        return rmseNN

    def testOLS(self, y=None, X=None):
        if y is None:
            y = self.df.loc[self.ntraining:self.endIndex-1, self.endog].values
            X = self.df.loc[self.ntraining:self.endIndex-1, self.exog].values
        Xols = sm.add_constant(X, has_constant="add")
        yhatOls = self.ols.predict(exog=Xols)
        rmseOLS = np.sqrt(np.mean((y - yhatOls) ** 2))
        return rmseOLS

    def trainingDatasetTestNN(self):
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        return self.testNN(y=y, X=X)

    def trainingDatasetTestOLS(self):
        y = self.df.loc[self.beginIndex:self.ntraining, self.endog].values
        X = self.df.loc[self.beginIndex:self.ntraining, self.exog].values
        return self.testOLS(y=y, X=X)

    def plot(self, epochs, trainError, testError, olsErrorTrain, olsErrorTest):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(epochs, trainError, label="NN Training RMSE")
        axs.plot(epochs, testError, label="NN Testing RMSE")
        axs.axhline(y=olsErrorTrain, color='r', linestyle='dashed', label="OLS Training RMSE")
        axs.axhline(y=olsErrorTest, color='g', linestyle='dashdot', label="OLS Testing RMSE")
        axs.set(title="Selecting Training Epochs for Deep Neural Network")
        axs.legend()
        axs.grid()
        axs.set_xlabel("Epochs")
        axs.set_ylabel("RMSE")
        plt.savefig(os.path.join(self.dirname, f"AssetReturn_{self.security}.jpeg"),
                    dpi=500)
        plt.show()

    def findOptimalTrainingEpochs(self):
        epochs = list(range(10, self.maxEpochs, 10))
        testError = []
        trainError = []
        self.fitOLS()
        olsErrorTrain = self.trainingDatasetTestOLS()
        olsErrorTest = self.testOLS()
        for epoch in epochs:
            nnerror = self.fit(nepochs=epoch)
            self.logger.info("Epoch: %d, Fitting RMSE: %f", epoch, nnerror)
            nnErrorTrain = self.trainingDatasetTestNN()
            nnErrorTest = self.testNN()
            testError.append(nnErrorTest)
            trainError.append(nnErrorTrain)
        self.plot(epochs, trainError, testError, olsErrorTrain, olsErrorTest)
        self.logger.info("OLS RMS error on training dataset: %f, testing dataset: %f", olsErrorTrain, olsErrorTest)


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    pred = AssetReturnPredictor(dirname, "SPY")
    np.random.seed(32)
    tf.random.set_seed(32)
    pred.findOptimalTrainingEpochs()