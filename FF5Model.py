import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging
import os
import statsmodels.api as sm

logging.basicConfig(level=logging.DEBUG)


class FF5FactorBase(ABC):
    PRICE_COL = "Close"
    PERIOD = 21

    def __init__(self, dirname, ff5FactorFile, security, trainTestSplit=0.9):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dirname = dirname
        self.security = security
        dfFF5 = pd.read_csv(os.path.join(dirname, ff5FactorFile), parse_dates=["Date"])
        dfSecurity = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        self.endog, self.exog = None, None
        self.df = self.processColumns(dfFF5, dfSecurity)
        self.nTrainRows = int(self.df.shape[0] * trainTestSplit)
        self.model = None

    def processColumns(self, dfFF5, dfSecurity):
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        for col in cols:
            dfFF5.loc[:, col] = dfFF5.loc[:, col].astype(np.float32)
        dfSecurity.loc[:, self.PRICE_COL] = dfSecurity.loc[:, self.PRICE_COL].astype(np.float32)
        price = dfSecurity.loc[:, self.PRICE_COL].values
        return1Mo = price[self.PERIOD:] / price[0:-self.PERIOD] - 1
        dfSecurity.loc[:, "1MoReturn"] = 0
        dfSecurity.loc[self.PERIOD:, "1MoReturn"] = return1Mo
        dfMerged = pd.merge(dfSecurity, dfFF5, on=["Date"], how="inner")
        self.endog = "1MoRetMinusRF"
        dfMerged.loc[:, self.endog] = dfMerged.loc[:, "1MoReturn"] - dfMerged.loc[:, "RF"]

        self.exog = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        dfMerged = dfMerged.loc[self.PERIOD:, ["Date", self.endog] + self.exog].reset_index(drop=True)
        return dfMerged

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Sub class {self.__class__.__name__} needs to implement")

    def test(self):
        groundTruth = self.df.loc[self.nTrainRows:, self.endog].values
        exog = self.df.loc[self.nTrainRows:, self.exog].values
        exog = sm.add_constant(exog, has_constant="add")
        predicted = self.model.predict(exog)
        diff = predicted - groundTruth
        mse = np.sqrt(np.dot(diff, diff) / diff.shape[0])
        self.logger.info("MSE error on test dataset: %f", mse)


class OLSModel(FF5FactorBase):
    def fit(self):
        endog = self.df.loc[0:self.nTrainRows, self.endog].values
        exog = self.df.loc[0:self.nTrainRows, self.exog].values
        exog = sm.add_constant(exog, has_constant="add")
        self.model = sm.OLS(endog, exog).fit()
        self.logger.info(self.model.summary())
        summaryfile = os.path.join(self.dirname, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(self.model.summary().as_text())


class BayesianLinearRegModel(FF5FactorBase):
    def fit(self):
        endog = self.df.loc[0:self.nTrainRows, self.endog].values
        exog = self.df.loc[0:self.nTrainRows, self.exog].values
        exog = sm.add_constant(exog, has_constant="add")
        self.model = sm.OLS(endog, exog).fit_regularized(alpha=1.0, L1_wt=0.0)


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    ff5file = "ff5Factors.csv"
    security = "BAC"
    ols = OLSModel(dirname, ff5file, security)
    ols.fit()
    ols.test()

    bayesian = BayesianLinearRegModel(dirname, ff5file, security)
    bayesian.fit()
    bayesian.test()