import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


class TaylorRule:
    def __init__(self, dirname, effrFile, inflationFile, outputGapFile, inflationTarget=0.02,
                 beginDate="1980-01-01", trainTestSplit=0.9):
        self.dirname = dirname
        self.trainTestSplit = trainTestSplit
        self.logger = logging.getLogger(self.__class__.__name__)
        effr = pd.read_csv(os.path.join(dirname, f"{effrFile}.csv"), parse_dates=["DATE"])
        effr.rename(columns={"DFF": "intRate"}, inplace=True)
        self.convertColToFloat(effr, "intRate", divideBy=100)
        effr = self.convertToQuarterly(effr, ["intRate"])
        self.beginDate = pd.to_datetime(beginDate)
        self.endog = "intRate"

        inflation = pd.read_csv(os.path.join(dirname, f"{inflationFile}.csv"), parse_dates=["DATE"])
        inflation.rename(columns={inflationFile: "pi"}, inplace=True)
        self.convertColToFloat(inflation, "pi", divideBy=100)
        inflation.loc[:, "pi_m_target"] = inflation.loc[:, "pi"] - inflationTarget

        df = pd.merge(effr, inflation, on=["DATE"], how="inner")

        outputGap = pd.read_csv(os.path.join(dirname, f"{outputGapFile}.csv"), parse_dates=["DATE"])
        outputGap.rename(columns={"GDPC1_GDPPOT": "output_gap"}, inplace=True)
        self.convertColToFloat(outputGap, "output_gap", divideBy=100)
        df = pd.merge(df, outputGap, on=["DATE"], how="inner")

        df = df.loc[df.DATE >= self.beginDate, :].reset_index(drop=True)

        self.exog = ["pi_m_target", "output_gap"]
        self.df = df

    def convertColToFloat(self, df, col, divideBy=1.0):
        if (df.loc[:, col] == ".").sum() > 0:
            df.drop(np.where(df.loc[:, col] == ".")[0], inplace=True)
        df.loc[:, col] = df.loc[:, col].astype(np.float64) / divideBy
        df.reset_index(drop=True, inplace=True)

    def convertToQuarterly(self, df, cols):
        df.loc[:, 'quarter'] = ((df.DATE.dt.month.values - 1) // 3)
        df.loc[:, 'year'] = df.DATE.dt.year
        ypart = df[['year', 'quarter'] + cols]
        ypart = ypart.groupby(['year', 'quarter']).mean().reset_index(drop=False)
        ydate = df[["DATE", "year", "quarter"]].groupby(["year", "quarter"]).first().reset_index(drop=False)
        ydate.loc[:, "DATE"] = ydate.DATE + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(-1)
        df = pd.merge(ydate, ypart, on=["year", "quarter"], how="inner")
        df.drop(columns=["year", "quarter"], inplace=True)
        return df

    def trainData(self):
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        y = self.df.loc[0:ntrain, self.endog].values - self.df.loc[0:ntrain, "pi"].values
        X = self.df.loc[:ntrain, self.exog].values
        return y, X

    def fitOLS(self):
        y, X = self.trainData()
        X = sm.add_constant(X, has_constant="add")
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
        ax.plot(date, y + self.df.loc[0:ntrain, "pi"].values, label="Actual")
        yPred = self.olsModel.fittedvalues
        ax.plot(date, yPred + self.df.loc[0:ntrain, "pi"].values, label="OLS Predicted")
        yPred = self.markovModel.fittedvalues
        ax.plot(date, yPred + self.df.loc[0:ntrain, "pi"].values, label="Regime Switch Predicted")
        ax.legend()
        ax.set(title="Effective Federal Funds Rate (EFFR)")
        ax.grid()
        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"effr_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

    def fitRegimeSwitch(self):
        y, X = self.trainData()
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        date = self.df.loc[0:ntrain, "DATE"].values
        np.random.seed(1024)
        self.markovModel = sm.tsa.MarkovRegression(endog=y, k_regimes=3, trend='c', exog=X,
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
        ax.set(title="Probability of regime 1 (Low EFFR)")
        ax.grid()

        ax = axes[1]
        ax.plot(date, self.markovModel.filtered_marginal_probabilities[:, 1])
        ax.set(title="Probability of regime 2 (Medium EFFR)")
        ax.grid()

        ax = axes[2]
        ax.plot(date, self.markovModel.filtered_marginal_probabilities[:, 2])
        ax.set(title="Probability of regime 3 (High EFFR)")
        ax.grid()

        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"regime_prob_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

    def fit(self):
        self.fitOLS()
        self.fitRegimeSwitch()
        self.plotTrainingFit()


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    trule = TaylorRule(dirname, "DFF", "A191RI1Q225SBEA", "fredgraph_OutputGap", trainTestSplit=1.0)
    trule.fit()