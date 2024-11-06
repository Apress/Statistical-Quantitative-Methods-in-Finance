import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import logging


DATADIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
PLOTDIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\plots"
logging.basicConfig(level=logging.DEBUG)


class GammaGLM(object):
    PERIOD = 5

    def __init__(self, endog, exog, trainPerc=0.9):
        filename = os.path.join(DATADIR, endog + ".csv")
        y = pd.read_csv(filename, parse_dates=["DATE"])
        self.convertColumnToFloat(y, endog)
        self.plotEndog(y, endog)
        y.loc[:, 'quarter'] = ((y.DATE.dt.month.values - 1) // 3)
        y.loc[:, 'year'] = y.DATE.dt.year
        ypart = y[['year', 'quarter', endog]]
        ypart = ypart.groupby(['year', 'quarter']).sum().reset_index(drop=False)
        y.drop(columns=[endog], inplace=True)
        y = pd.merge(y, ypart, on=['year', 'quarter'], how="left")
        self.beginIndex = None
        self.endIndex = None
        self.origEndog = endog
        for xi in exog:
            filename = os.path.join(DATADIR, xi + ".csv")
            x = pd.read_csv(filename, parse_dates=["DATE"])
            x.loc[:, "MonthBegin"] = x.DATE + pd.offsets.MonthBegin(0)
            x = x.groupby(["MonthBegin"]).first().reset_index(drop=False)
            x.DATE = x.MonthBegin
            y = pd.merge(y, x, how="inner", on=["DATE"])
            self.convertColumnToFloat(y, xi)
        y = self.calculateEndogExogVars(y, endog, exog)
        self.df = y

        self.testdata = int(trainPerc * (self.endIndex - self.beginIndex)) + self.beginIndex
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def plotEndog(self, df, endog):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        y = df.loc[:, endog].values
        dates = df.DATE.values
        axs.plot(dates, y)
        axs.grid()
        axs.set_title("Non-Farm Payroll Employment")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "nonFarmPayroll.jpeg"),
                    dpi=500)

    def calculateEndogExogVars(self, df, endog, exog):
        vals = df.loc[:, endog].values
        yvar = "AvgNonFarmPayrollPerQtr"
        df.loc[:, yvar] = 0.0
        for i in range(3, df.shape[0]):
            df.loc[i - 3, yvar] = vals[i-3:i+1].mean()
        self.endog = yvar

        # 3-year percent change in real GDP, SP500
        exog_cols = []
        for x in exog:
            col = x + "_3year_perc_change"
            rval = df.loc[:, x].values
            rval = np.where(rval == 0, 1E-8, rval)
            change = rval[11:] / rval[0:-11] - 1
            df.loc[:, col] = 0
            df.loc[11:, col] = change
            exog_cols.append(col)

        self.exog = exog_cols
        self.nvars = len(self.exog)
        self.beginIndex = 12
        self.endIndex = df.shape[0] - 12
        return df

    def convertColumnToFloat(self, df, col):
        if (df.loc[:, col] == ".").sum() > 0:
            df.drop(np.where(df.loc[:, col] == ".")[0], inplace=True)
            df.loc[:, col] = df.loc[:, col].astype(np.float64)
            df.reset_index(drop=True, inplace=True)

    def fit(self):
        y = self.df.loc[self.beginIndex:self.testdata, self.endog].values
        X = self.df.loc[self.beginIndex:self.testdata, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        glm = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.inverse_power()))
        glm = glm.fit()
        self.logger.info(glm.summary(xname=['constant'] + self.exog))
        summaryfile = os.path.join(PLOTDIR, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(glm.summary(xname=['constant'] + self.exog).as_text())
        self.model = glm

    def plotResid(self):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        yendog = self.model.model.endog
        yhatv = self.model.predict(self.model.model.exog)

        dates = self.df.loc[self.beginIndex:self.testdata, "DATE"].values

        axs.plot(dates, yendog, label="y")
        axs.plot(dates, yhatv, "-.", label="ypred")
        axs.grid()
        axs.legend()
        axs.set_title("Predicted vs. Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, f"trainResid{self.__class__.__name__}.jpeg"),
                    dpi=500)

    def tabulateTestResults(self):
        # calculate the month when non-farm payroll reaches or exceeds 300K
        yvar = "MonthPayRollGt300K"
        ypredvar = "PredMonthPayRollGt300K"
        df = self.df
        month = df.loc[:, "DATE"].dt.month
        begin_month = np.where(month.values == 1)
        df.loc[:, yvar] = 0.0
        df.loc[:, ypredvar] = 0
        endog = df.loc[:, self.origEndog].values
        month_indx = []

        for i in begin_month[0]:
            if (i > self.testdata) and (df.loc[i, self.endog] != 0.0):
                month_indx.append(i)
                total = np.cumsum(endog[i:i+4])
                df.loc[i, yvar] = np.where(total >= 300000)[0][0]
                df.loc[i, ypredvar] = int(300000 / df.loc[i, "ypred"])
                self.logger.info("Date: %s, predicted quarterly non-farm payroll: %f, actual: %f", df.loc[i, "DATE"],
                                 df.loc[i, "ypred"], df.loc[i, self.endog])

        dates = df.loc[month_indx, "DATE"].values
        yval = df.loc[month_indx, yvar].values
        yvalhat = df.loc[month_indx, ypredvar].values
        prollhat = df.loc[month_indx, "ypred"].values
        proll = df.loc[month_indx, self.endog].values

        df = pd.DataFrame({"Date": dates, "Predicted Quarter": yvalhat,
                           "Actual Quarter": yval,
                           "Predicted Qtr Payroll": prollhat,
                           "Actual QtrAvg Payroll": proll})
        df.loc[:, "Perc Diff"] = df.loc[:, "Predicted Qtr Payroll"] / df.loc[:, "Actual QtrAvg Payroll"] - 1

        df.to_csv(os.path.join(PLOTDIR, f"{self.__class__.__name__}.csv"), index=False)
        self.logger.info(df.to_latex(index=False, float_format="{:.2f}".format))
        self.logger.info(df)

    def test(self):
        testdata = self.testdata + 1
        X = self.df.loc[testdata:, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        ypred = self.model.predict(X)
        self.df.loc[:, "ypred"] = 0.0
        self.df.loc[testdata:, "ypred"] = ypred
        self.tabulateTestResults()


if __name__ == "__main__":
    glm = GammaGLM("PAYEMS", ["GDPC1", "SP500"])
    # PAYEMS: Nonfarm payroll (monthly), seasonally adjusted
    # CCLACBW027SBOG: Loan on credit card and other revolving plans (weekly) 200
    # PCEPI: PCE price index (monthly) 100
    # DSPIC96: Real disposable income (monthly) 2000
    # MORTGAGE30US: 30 year mortgage rate (weekly) 100

    glm.fit()
    glm.plotResid()
    glm.test()
