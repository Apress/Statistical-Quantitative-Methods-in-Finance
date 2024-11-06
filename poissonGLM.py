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


class PoissonGLM(object):
    def __init__(self, endogName, exogNames, trainPerc=0.9):
        filename = os.path.join(DATADIR, endogName + ".csv")
        y = pd.read_csv(filename, parse_dates=["DATE"])
        y.loc[:, "year"] = y.DATE.dt.year
        y.loc[:, "month"] = y.DATE.dt.month
        y.loc[:, "quarter"] = (y.month.values - 1) // 3
        self.convertColumnToFloat(y, endogName)
        for xi in exogNames:
            filename = os.path.join(DATADIR, xi + ".csv")
            x = pd.read_csv(filename, parse_dates=["DATE"])
            x.loc[:, "year"] = x.DATE.dt.year
            x.loc[:, "month"] = x.DATE.dt.month
            x.loc[:, "quarter"] = (x.month.values - 1) // 3
            self.convertColumnToFloat(x, xi)
            x = x[["year", "quarter", xi]].groupby(["year", "quarter"]).mean().reset_index(drop=False)
            y = pd.merge(y, x, on=["year", "quarter"], how="inner")

        y.replace(".", np.nan, inplace=True)
        floatcols = set(exogNames + [endogName])
        for col in y.columns:
            if col in floatcols:
                y.loc[:, col] = y.loc[:, col].astype(np.float64)
        y.ffill(inplace=True)
        self.endog = endogName
        self.exog = exogNames
        y = self.calculateTransformedVars(y)
        self.df = y

        self.testdata = int(trainPerc * self.df.shape[0]) - 1
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def convertColumnToFloat(self, df, col):
        df.loc[:, col] = df.loc[:, col].replace(".", np.nan).astype(np.float64).ffill()

    def calculateDiffOverAvg(self, df, col, lag, newcolname):
        vals = df.loc[:, col].values
        avg = np.zeros(vals.shape[0], dtype=np.float64)
        for i in range(lag):
            avg[i] = vals[0:i+1].sum() / (i+1)

        sumv = vals[0:lag].sum()
        for i in range(lag, vals.shape[0]):
            sumv += vals[i] - vals[i-lag]
            avg[i] = sumv/lag
        df.loc[:, newcolname] = vals/avg - 1
        return df

    def calculateTransformedVars(self, y):
        # convert credit card delinq rate to decimal
        y.loc[:, self.endog] = y.loc[:, self.endog] / 100.0

        # calculate GDP growth rate
        y.sort_values(by=["year", "quarter"], inplace=True)
        gdp = y.loc[:, "GDP"].values
        growthRate = gdp[1:] / gdp[0:-1] - 1
        y.loc[:, "GDPGrowthRate"] = 0.0
        y.loc[1:, "GDPGrowthRate"] = growthRate

        # convert TERMCBCCALLNS: CB interest rate on credit cards (monthly) to decimal
        y.loc[:, "TERMCBCCALLNS"] = y.TERMCBCCALLNS / 100.0

        # calculate int rate - trailing 8 quarter (2 year) average
        y = self.calculateDiffOverAvg(y, "TERMCBCCALLNS", 8, "IntRateDiff")

        # divide CCLACBW027SBOG: Loan on credit card and other revolving plans (weekly) by 200
        y.loc[:, "CCLACBW027SBOG"] = y.CCLACBW027SBOG / 200.0

        # divide PCEPI: PCE price index (monthly) by 100
        y.loc[:, "PCEPI"] = y.PCEPI / 100.0
        y = self.calculateDiffOverAvg(y, "PCEPI", 8, "InflDiff")

        # divide DSPIC96: Real disposable income (monthly) by 2000
        y.loc[:, "DSPIC96"] = y.DSPIC96 / 2000.0
        y = self.calculateDiffOverAvg(y, "DSPIC96", 8, "RealDispIncDiff")

        # convert MORTGAGE30US: 30 year mortgage rate (weekly) to decimal
        y.loc[:, "MORTGAGE30US"] = y.MORTGAGE30US / 100.0
        y = self.calculateDiffOverAvg(y, "MORTGAGE30US", 8, "Mort30Diff")

        y.loc[:, "BeforeGFC"] = np.where(y.year < 2010 , 1, 0)

        # divide y by normalized credit card outstanding loans
        y.loc[:, self.endog] = y.loc[:, self.endog] / y.loc[:, "CCLACBW027SBOG"]

        self.exog = ["GDPGrowthRate", "IntRateDiff", "InflDiff", "RealDispIncDiff", "Mort30Diff",
                     "BeforeGFC"]
        return y

    def fit(self):
        y = self.df.loc[8:self.testdata, self.endog].values
        X = self.df.loc[8:self.testdata, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        glm = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.log()))
        glm = glm.fit()
        self.logger.info(glm.summary())
        summaryfile = os.path.join(PLOTDIR, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(glm.summary().as_text())
        self.model = glm

    def plotResid(self):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        yendog = self.model.model.endog
        yhatv = self.model.predict(self.model.model.exog)
        resid = np.log(yendog
                       /yhatv)
        self.logger.info("mu = %f, sd = %f", self.df.loc[:, self.endog].mean(), self.df.loc[:, self.endog].std())

        dates = self.df.loc[8:self.testdata, "DATE"].values

        axs.plot(dates, resid)
        axs.grid()
        axs.set_title("Residual Plot (Training Dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "trainResidPoisson.jpeg"),
                    dpi=500)

    def plotTestResults(self, y, ypred):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        resid = np.log(y / ypred)

        dates = self.df.loc[self.testdata+1:, "DATE"].values
        axs[0].plot(dates, resid)
        axs[0].plot()
        axs[0].grid()
        axs[0].set_title("Residual Plot (Test Dataset)")
        axs[1].plot(dates, y, label="y")
        axs[1].plot(dates, ypred, "-.", label="ypred")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title("Predicted vs. Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "testResidPoisson.jpeg"),
                    dpi=500)

    def test(self):
        testdata = self.testdata + 1
        y = self.df.loc[testdata:, self.endog].values
        X = self.df.loc[testdata:, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        ypred = self.model.predict(X)
        self.plotTestResults(y, ypred)


if __name__ == "__main__":
    poisson = PoissonGLM("DRCCLACBS", ["TERMCBCCALLNS", "CCLACBW027SBOG", "PCEPI", "GDP",
                                       "DSPIC96", "MORTGAGE30US"])
    # TERMCBCCALLNS: CB interest rate on credit cards (monthly)  100
    # CCLACBW027SBOG: Loan on credit card and other revolving plans (weekly) 200
    # PCEPI: PCE price index (monthly) 100
    # DSPIC96: Real disposable income (monthly) 2000
    # MORTGAGE30US: 30 year mortgage rate (weekly) 100

    poisson.fit()
    poisson.plotResid()
    poisson.test()
