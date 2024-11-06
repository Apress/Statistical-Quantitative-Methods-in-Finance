import numpy as np
import logging
import statsmodels.api as sm
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


class Heckman2StepModel(object):
    """ Tobit-II (censored) regression model using Heckman 2-step approach """

    def __init__(self, endog: np.ndarray, exog: np.ndarray, low=None, high=None,
                 include_constant=True, train_test_ratio=0.9, low_threshold=0.15,
                 high_threshold=0.15):
        """
        Initialize the regression model
        :param endog: y
        :param exog: X
        :param low: Low threshold for censoring
        :param high: High threshold for censoring
        :param include_constant:
        :param train_test_ratio:
        """
        assert (low is not None) or (high is not None), "both low and high cannot be None"
        if (low is not None) and (high is not None):
            assert low < high, "low must be strictly less than high"
        self.low = low
        self.high = high
        self.lowProbitModel = None
        self.highProbitModel = None
        self.olsModel = None
        self.endog = endog
        self.exog = exog
        self.includeConstant = include_constant
        self.trainTestRatio = train_test_ratio
        self.lowThreshold = low_threshold
        self.highThreshold = high_threshold
        self.ntraining = int(self.trainTestRatio * self.endog.shape[0])
        self.logger = logging.getLogger(self.__class__.__name__)

    def fitProbit(self, endog, exog, threshold):
        rows = np.where(endog >= threshold, 1, 0)
        model = sm.Probit(rows, exog)
        return model.fit()

    def fitOLS(self, endog, exog):
        model = sm.OLS(endog, exog)
        return model.fit()

    def fit(self):
        """
        Fit the Heckman regression model to the data
        """
        ntraining = self.ntraining
        exog = self.exog
        if self.includeConstant:
            exog = sm.add_constant(self.exog, has_constant="add")
        olsFlag = np.ones(ntraining, dtype=bool)
        if self.low is not None:
            olsFlag[self.endog[0:ntraining] <= self.low] = False
            self.lowProbitModel = self.fitProbit(self.endog[0:ntraining], exog[0:ntraining, :], self.low)
        if self.high is not None:
            olsFlag[self.endog[0:ntraining] >= self.high] = False
            self.highProbitModel = self.fitProbit(self.endog[0:ntraining], exog[0:ntraining, :], self.high)
        self.olsModel = self.fitOLS(self.endog[0:ntraining][olsFlag], exog[0:ntraining, :][olsFlag, :])

    def predict(self, exog: np.ndarray = None) -> np.ndarray:
        """
        Predict the output of the model using exogeneous variables as input.
        :param exog: exogeneous variables (X)
        :return: output value from the model (y)
        """
        if exog is None:
            exog = self.exog[self.ntraining:, :]
        if self.includeConstant:
            exog = sm.add_constant(exog, has_constant="add")
        result = np.zeros(exog.shape[0], dtype=np.float64)
        olsFlag = np.ones(exog.shape[0], dtype=bool)
        if self.low is not None:
            lowProb = self.lowProbitModel.predict(exog)
            lowVals = lowProb < (1 - self.lowThreshold)
            olsFlag[lowVals] = False
            result[lowVals] = self.low
        if self.high is not None:
            highProb = self.highProbitModel.predict(exog)
            highVals = (highProb > self.highThreshold)
            olsFlag[highVals] = False
            result[highVals] = self.high
        result[olsFlag] = self.olsModel.predict(exog[olsFlag, :])
        return result

    @staticmethod
    def rmse(y1, y2):
        return np.sqrt(np.mean((y1 - y2)**2))


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    df = pd.read_csv(os.path.join(dirname, "student_scores.csv"))
    df.loc[:, "prog_codes"] = df.loc[:, "prog"].astype("category").cat.codes
    high = 800
    y = df.loc[:, "apt"].values
    X = df.loc[:, ["read","math","prog_codes"]].values
    heckman = Heckman2StepModel(y, X, high=high)
    heckman.fit()
    predicted = heckman.predict()
    actual = y[heckman.ntraining:]
    rmse1 = Heckman2StepModel.rmse(actual, predicted)

    # fit an OLS model
    Xconst = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y[0:heckman.ntraining], Xconst[0:heckman.ntraining, :])
    ols = ols.fit()
    olsPred = ols.predict(Xconst[heckman.ntraining:, :])
    rmse2 = Heckman2StepModel.rmse(actual, olsPred)

    logging.info("RMSE from Heckman model: %f, OLS model: %f", rmse1, rmse2)

    # plot
    predictors = ["Heckman"] * predicted.shape[0] + ["OLS"] * olsPred.shape[0] + ["Actual"] * actual.shape[0]
    values = np.concatenate((predicted, olsPred, actual), axis=0)
    ids = df.loc[heckman.ntraining:, "id"].values
    ids = np.concatenate((ids, ids, ids))
    df = pd.DataFrame({"Id": ids, "Aptitude Score": values, "Predictor": predictors})
    sns.lineplot(data=df, x="Id", y="Aptitude Score", hue="Predictor")
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig(os.path.join(dirname, f"heckman_v_ols.jpeg"),
                dpi=500)
    plt.show()

