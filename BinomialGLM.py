import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import logging
import scipy.stats as ss


DATADIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
PLOTDIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\plots"
logging.basicConfig(level=logging.DEBUG)


class BinomialGLM(object):
    PERIOD = 5

    def __init__(self, security, rates, trainPerc=0.9):
        filename = os.path.join(DATADIR, security + ".csv")
        y = pd.read_csv(filename, parse_dates=["DATE"])
        self.convertColumnToFloat(y, security)
        for xi in rates:
            filename = os.path.join(DATADIR, xi + ".csv")
            x = pd.read_csv(filename, parse_dates=["DATE"])
            y = pd.merge(y, x, how="left", on=["DATE"])
            self.convertRateToFloat(y, xi)
        y = self.calculateEndogExogVars(y, security, rates)
        self.df = y

        self.testdata = int(trainPerc * self.df.shape[0]) - 1
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def calculateEndogExogVars(self, y, col, rates):
        vals = y.loc[:, col].values
        ret = vals[1+self.PERIOD:]/vals[1:-self.PERIOD] - 1

        y.loc[:, "positive_ret"] = 0
        self.endIndex = y.shape[0] - self.PERIOD - 2
        y.loc[0:self.endIndex, "positive_ret"] = np.where(ret > 0, 1.0, 0.0)

        self.endog = "positive_ret"
        y.loc[:, "lastret"] = 0
        y.loc[2+self.PERIOD:, "lastret"] = ret[0:-1]
        y.loc[:, "indicator"] = np.where(y.lastret.values > 0.02, 1.0, 0.0)

        ma3 = self.movingAverage(vals, 3)
        ma10 = self.movingAverage(vals, 10)
        y.loc[:, "ma3_10"] = 0.0
        y.loc[11:, "ma3_10"] = np.where(ma3[10:-1] > ma10[10:-1], 1.0, 0.0)

        vol21day = self.volatility(ret, 21)
        vol1yr = self.volatility(ret, 252)
        y.loc[:, "vol21_252"] = 0
        y.loc[253+self.PERIOD:, "vol21_252"] = np.where(vol21day[252:] > vol1yr[252:], 1.0, 0.0)

        # percent change in interest rate over the period
        rate_change_cols = []
        for rate in rates:
            col = rate + "_change"
            rval = y.loc[:, rate].values
            rval = np.where(rval == 0, 1E-8, rval)
            change = rval[self.PERIOD:] / rval[0:-self.PERIOD] - 1
            y.loc[:, col] = 0
            y.loc[1+self.PERIOD:, col] = change[0:-1]
            rate_change_cols.append(col)

        self.exog = ["indicator", "ma3_10", "vol21_252"] + rate_change_cols
        self.nvars = len(self.exog)
        self.beginIndex = 253
        return y

    def movingAverage(self, arr, period):
        res = np.zeros(arr.shape[0], dtype=np.float64)
        sumval = np.sum(arr[0:period])
        for i in range(period, arr.shape[0]):
            res[i] = sumval / period
            sumval += arr[i] - arr[i-period]

        return res

    def volatility(self, arr, period):
        res = np.zeros(arr.shape[0], dtype=np.float64)
        sumval = np.sum(arr[0:period])
        sumsq = np.dot(arr[0:period], arr[0:period])
        for i in range(period, arr.shape[0]):
            res[i] = np.sqrt(sumsq/period - (sumval/period)**2)
            sumval += arr[i] - arr[i-period]
            sumsq += arr[i]*arr[i] - arr[i-period]*arr[i-period]
        return res

    def convertRateToFloat(self, df, col):
        df.loc[:, col] = df.loc[:, col].replace(".", np.nan).astype(np.float64).ffill()
        df.loc[:, col] = df.loc[:, col] / 100.0  # convert to decimal

    def convertColumnToFloat(self, df, col):
        if (df.loc[:, col] == ".").sum() > 0:
            df.drop(np.where(df.loc[:, col] == ".")[0], inplace=True)
            df.loc[:, col] = df.loc[:, col].astype(np.float64)
            df.reset_index(drop=True, inplace=True)

    def fit(self):
        y = self.df.loc[self.beginIndex:self.testdata, self.endog].values
        X = self.df.loc[self.beginIndex:self.testdata, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        glm = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
        glm = glm.fit()
        self.logger.info(glm.summary(xname=['constant'] + self.exog))
        summaryfile = os.path.join(PLOTDIR, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(glm.summary(xname=['constant'] + self.exog).as_text())
        self.model = glm

    def plotResid(self):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        yendog = self.model.model.endog
        yhatv = self.model.predict(self.model.model.exog)
        resid = yendog - yhatv

        dates = self.df.loc[self.beginIndex:self.testdata, "DATE"].values

        axs[0].plot(dates, resid)
        axs[0].grid()
        axs[0].set_title("Residual Plot (Training Dataset)")
        axs[1].hist(yhatv, bins=40, density=True)
        axs[1].grid()
        axs[1].set_title("Histogram of Residuals")
        axs[2].plot(dates, yendog, label="y")
        axs[2].plot(dates, yhatv, "-.", label="ypred")
        axs[2].grid()
        axs[2].legend()
        axs[2].set_title("Predicted vs. Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "trainResidBinomial.jpeg"),
                    dpi=500)

    def plotTestResults(self, y, ypred):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        resid = (y - ypred)

        dates = self.df.loc[self.testdata+1:self.endIndex, "DATE"].values
        axs[0].hist(resid, bins=40, density=True)
        axs[0].grid()
        axs[0].set_title("Histogram of Residuals")
        axs[1].plot(dates, y, label="y")
        axs[1].plot(dates, ypred, "-.", label="ypred")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title("Predicted vs. Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "testResidBinomial.jpeg"),
                    dpi=500)

    def test(self):
        testdata = self.testdata + 1
        y = self.df.loc[testdata:self.endIndex, self.endog].values
        X = self.df.loc[testdata:self.endIndex, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        ypred = self.model.predict(X)
        self.plotTestResults(y, ypred)


if __name__ == "__main__":
    glm = BinomialGLM("SP500", ["DGS1MO"])
    # TERMCBCCALLNS: CB interest rate on credit cards (monthly)  100
    # CCLACBW027SBOG: Loan on credit card and other revolving plans (weekly) 200
    glm.fit()
    glm.plotResid()
    glm.test()
