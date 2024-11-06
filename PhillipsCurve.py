import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


class PhillipsCurve:
    def __init__(self, dirname, inflationFile, expectedInflFile, unemploymentFile, naturalUnempFile,
                 trainTestSplit=0.9):
        self.dirname = dirname
        self.trainTestSplit = trainTestSplit
        self.logger = logging.getLogger(self.__class__.__name__)

        inflation = pd.read_csv(os.path.join(dirname, f"{inflationFile}.csv"), parse_dates=["DATE"])
        self.convertColToFloat(inflation, inflationFile)
        pceValues = inflation.loc[:, inflationFile].values
        inflationVal = pceValues[12:] / pceValues[0:-12] - 1
        inflation.loc[:, "pi"] = 0
        inflation.loc[12:, "pi"] = inflationVal
        inflation.loc[:, "lagged_pi"] = 0
        inflation.loc[13:, "lagged_pi"] = inflationVal[0:-1]
        self.endog = "pi"

        expInflation = pd.read_csv(os.path.join(dirname, f"{expectedInflFile}.csv"), parse_dates=["DATE"])
        expInflation.rename(columns={expectedInflFile: "E_pi"}, inplace=True)
        self.convertColToFloat(expInflation, "E_pi", divideBy=100)
        expInflation = self.convertToMonthly(expInflation, ["E_pi"])
        df = pd.merge(inflation, expInflation, on=["DATE"], how="inner")

        unemp = pd.read_csv(os.path.join(dirname, f"{unemploymentFile}.csv"), parse_dates=["DATE"])
        unemp.rename(columns={unemploymentFile: "unemp"}, inplace=True)
        self.convertColToFloat(unemp, "unemp", divideBy=100)
        df = pd.merge(df, unemp, on=["DATE"], how="inner")

        nairu = pd.read_csv(os.path.join(dirname, f"{naturalUnempFile}.csv"), parse_dates=["DATE"])
        nairu.rename(columns={naturalUnempFile: "nairu"}, inplace=True)
        self.convertColToFloat(nairu, "nairu", divideBy=100)
        self.addYearAndQuarterColumn(df)
        self.addYearAndQuarterColumn(nairu)
        nairu.drop(columns=["DATE"], inplace=True)
        df = pd.merge(df, nairu, on=["year", "quarter"], how="inner")
        df.loc[:, "u_m_un"] = df.loc[:, "unemp"] - df.loc[:, "nairu"]

        self.exog = ["E_pi", "lagged_pi", "u_m_un"]
        self.df = df

    def convertColToFloat(self, df, col, divideBy=1.0):
        if (df.loc[:, col] == ".").sum() > 0:
            df.drop(np.where(df.loc[:, col] == ".")[0], inplace=True)
        df.loc[:, col] = df.loc[:, col].astype(np.float64) / divideBy
        df.reset_index(drop=True, inplace=True)

    def convertToMonthly(self, df, cols):
        df.loc[:, 'year'] = df.DATE.dt.year
        df.loc[:, 'month'] = df.DATE.dt.month.values
        ypart = df[['year', 'month'] + cols]
        ypart = ypart.groupby(['year', 'month']).mean().reset_index(drop=False)
        ydate = df[["DATE", "year", "month"]].groupby(["year", "month"]).first().reset_index(drop=False)
        ydate.loc[:, "DATE"] = ydate.DATE + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(-1)
        df = pd.merge(ydate, ypart, on=["year", "month"], how="inner")
        df.drop(columns=["year", "month"], inplace=True)
        return df

    def addYearAndQuarterColumn(self, df):
        df.loc[:, 'quarter'] = ((df.DATE.dt.month.values - 1) // 3)
        df.loc[:, 'year'] = df.DATE.dt.year

    def trainData(self):
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        y = self.df.loc[0:ntrain, self.endog].values
        X = self.df.loc[:ntrain, self.exog].values
        return y, X

    def fitOLS(self):
        y, X = self.trainData()
        self.olsModel = sm.OLS(y, X)
        self.olsModel = self.olsModel.fit()
        self.logger.info(self.olsModel.summary())
        summaryfile = os.path.join(self.dirname, self.__class__.__name__ + "_ols.txt")
        with open(summaryfile, 'w') as fh:
            fh.write(self.olsModel.summary().as_text())

    def plotTrainingFit(self):
        y, X = self.trainData()
        fig, ax = plt.subplots(1, figsize=(10, 7))
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        date = self.df.loc[0:ntrain, "DATE"].values
        ax.plot(date, y, label="Actual")
        yPred = self.olsModel.fittedvalues
        ax.plot(date, yPred, label="OLS Predicted")
        yPred = self.markovModel.fittedvalues
        ax.plot(date, yPred, label="Regime Switch Predicted")
        ax.legend()
        ax.set(title="Inflation Predicted Using New Keynesian Hybrid Phillips Curve")
        ax.grid()
        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"infl_nkpc_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

    def fitRegimeSwitch(self):
        y, X = self.trainData()
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        date = self.df.loc[0:ntrain, "DATE"].values
        np.random.seed(1024)
        self.markovModel = sm.tsa.MarkovRegression(endog=y, k_regimes=3, trend='n', exog=X,
                                                   switching_trend=True,
                                                   switching_exog=True,
                                                   switching_variance=True)
        self.markovModel = self.markovModel.fit()

        self.logger.info(self.markovModel.summary())
        summaryfile = os.path.join(self.dirname, self.__class__.__name__ + "_regimeSwitch.txt")
        with open(summaryfile, 'w') as fh:
            fh.write(self.markovModel.summary().as_text())

        fig, axes = plt.subplots(3, figsize=(10, 7))

        ax = axes[0]
        ax.plot(date, self.markovModel.filtered_marginal_probabilities[:, 0])
        ax.set(title="Probability of regime 1")
        ax.grid()

        ax = axes[1]
        ax.plot(date, self.markovModel.filtered_marginal_probabilities[:, 1])
        ax.set(title="Probability of regime 2")
        ax.grid()

        ax = axes[2]
        ax.plot(date, self.markovModel.filtered_marginal_probabilities[:, 2])
        ax.set(title="Probability of regime 3")
        ax.grid()

        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"regime_prob_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

        fig, axes = plt.subplots(3, figsize=(10, 7))

        ax = axes[0]
        ax.plot(date, X[:, 0])
        ax.set(title="Expected Inflation")
        ax.grid()

        ax = axes[1]
        ax.plot(date, X[:, 1])
        ax.set(title="Lagged Inflation")
        ax.grid()

        ax = axes[2]
        ax.plot(date, X[:, 2])
        ax.set(title="Unemployment Above Natural Level (NAIRU)")
        ax.grid()

        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"regime_vars_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

    def fit(self):
        self.fitOLS()
        self.fitRegimeSwitch()
        self.plotTrainingFit()


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    nkpc_pc = PhillipsCurve(dirname, "PCEPI", "T10YIE", "UNRATE", "NROU", trainTestSplit=1.0)
    nkpc_pc.fit()