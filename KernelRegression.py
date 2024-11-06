import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.DEBUG)


class KernelRegression:
    def __init__(self, dirname, mktFile="SPY"):
        self.dirname = dirname
        self.sectors = { 'Communication services': 'XLC',
                         'Consumer discretionary': 'XLY',
                         'Consumer staples': 'XLP',
                         'Energy': 'XLE',
                         'Financials': 'XLF',
                         'Health care': 'XLV',
                         'Industrials': 'XLI',
                         'Materials': 'XLB',
                         'Real estate': 'XLRE',
                         'Technology': 'XLK',
                         'Utilities': 'XLU'
                         }
        self.mktFile = mktFile
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbolToEtf = {v:k for k, v in self.sectors.items()}
        self.dfs = {}
        self.mktDf = None
        self.variance = None
        self.readFiles()
        self.calculateEndogExogVars()

    def readFiles(self):
        for symbol in self.symbolToEtf.keys():
            self.dfs[symbol] = pd.read_csv(os.path.join(self.dirname, f"{symbol}.csv"), parse_dates=["Date"])
        self.mktDf = pd.read_csv(os.path.join(self.dirname, f"{self.mktFile}.csv"), parse_dates=["Date"])

    def calculateEndogExogVars(self):
        dfs = [self.mktDf] + list(self.dfs.values())
        for df in dfs:
            df.loc[:, "returns"] = 0
            price = df.loc[:, "Close"].values
            returns = price[1:] / price[0:-1] - 1
            df.loc[0:df.shape[0]-2, "returns"] = returns

    def calculateKernels(self, x, xi):
        multipliers = np.array([1.0/h for h in range(len(xi), 0, -1)])
        kernels = (multipliers / self.variance) * np.exp(-(((x - xi) / self.variance)**2)/2.0)
        normalizedKernels = kernels / kernels.sum()
        return normalizedKernels

    def calculateRMSE(self, actual, predicted):
        diff = (actual - predicted)
        return np.sqrt(np.sum(diff ** 2) / diff.shape[0])

    def calculateAdjustedR2(self, actual, predicted):
        diff = (actual - predicted)
        ssModel = np.sum(diff ** 2)
        avg = np.mean(actual)
        ssTotal = np.sum((actual - avg) ** 2)
        n = actual.shape[0]
        adjR2 = 1 - ((n-1)/(n-10-1)) * ssModel/ssTotal
        return adjR2

    def plot(self, actual, predicted, sector, begin, end):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        df = self.dfs[sector]
        dates = df.loc[begin:end, "Date"].values
        axs.plot(dates, actual, label="Actual")
        axs.plot(dates, predicted, label="Predicted")
        axs.grid()
        axs.legend()
        axs.set_xlabel("Date")
        axs.set_ylabel("Daily Return")
        axs.set(title=sector)
        plt.savefig(os.path.join(self.dirname, f"kernel_{sector}.jpeg"),
                    dpi=500)
        plt.show()

    def predict(self, beginDate, endDate):
        beginDate = pd.to_datetime(beginDate)
        endDate = pd.to_datetime(endDate)
        rmseList = []
        sectorList = []
        symbolList = []
        adjR2List = []
        for sector in self.symbolToEtf.keys():
            df = self.dfs[sector]
            begin = df.loc[df.Date == beginDate, :].index[0]
            end = df.loc[df.Date == endDate, :].index[0]
            beginMkt = self.mktDf.loc[self.mktDf.Date == beginDate, :].index[0]
            actual = df.loc[begin:end, "returns"].values
            predicted = np.zeros(actual.shape[0], dtype=np.float32)
            for j in range(begin, end+1, 1):
                beginIdx = beginMkt + j - begin
                mktRet = self.mktDf.loc[beginIdx, "returns"]
                self.variance = np.std(self.mktDf.loc[beginIdx-10:beginIdx, "returns"].values)
                prevMktReturns = self.mktDf.loc[beginIdx-10:beginIdx, "returns"].values
                prevSectorReturns = df.loc[j - 10:j, "returns"].values
                kernels = self.calculateKernels(mktRet, prevMktReturns)
                predicted[j-begin] = np.dot(prevSectorReturns, kernels)

            rmse = self.calculateRMSE(actual, predicted)
            adjr2 = self.calculateAdjustedR2(actual, predicted)
            self.plot(actual, predicted, sector, begin, end)
            self.logger.info("RMSE for sector %s: %f, adj R^2: %f", sector, rmse, adjr2)
            rmseList.append(rmse)
            sectorList.append(self.symbolToEtf[sector])
            symbolList.append(sector)
            adjR2List.append(adjr2)

        df = pd.DataFrame({"Sector": sectorList,
                           "Symbol": symbolList,
                           "RMSE": rmseList,
                           "Adj. R2": adjR2List})
        self.logger.info(df)


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\RLPy\data_merged\sectors"
    kernelReg = KernelRegression(dirname)
    beginDate = "2020-01-02"
    endDate = "2024-07-12"
    kernelReg.predict(beginDate, endDate)