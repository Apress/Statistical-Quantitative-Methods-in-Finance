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


class NormalGLM(object):
    def __init__(self, endogName, exogNames, trainPerc=0.9):
        filename = os.path.join(DATADIR, endogName + ".csv")
        y = pd.read_csv(filename, parse_dates=["DATE"])
        for xi in exogNames:
            filename = os.path.join(DATADIR, xi + ".csv")
            x = pd.read_csv(filename, parse_dates=["DATE"])
            y = pd.merge(y, x, on=["DATE"], how="inner")

        y.replace(".", np.nan, inplace=True)
        for col in y.columns:
            if col != "DATE":
                y.loc[:, col] = y.loc[:, col].astype(np.float64)
        y.ffill(inplace=True)
        self.endog = endogName
        self.exog = exogNames
        y = self.calculatePercChange(y)
        self.df = y

        self.testdata = int(trainPerc * self.df.shape[0]) - 1
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def calculatePercChange(self, y):
        yval = y.loc[:, self.endog].values
        ygrowth = yval[1:] - yval[0:-1]
        xgrowth = []
        for x in self.exog:
            xval = y.loc[:, x].values
            if x == "SP500":
                xgrowthi = xval[1:]/xval[0:-1] - 1
            else:
                xgrowthi = xval[1:] - xval[0:-1]
            xgrowth.append(xgrowthi)
        datadict = {"DATE": y.DATE[1:], self.endog: ygrowth}
        for i, x in enumerate(self.exog):
            datadict[x] = xgrowth[i]
        return pd.DataFrame(datadict)

    def fit(self):
        y = self.df.loc[:self.testdata, self.endog].values
        X = self.df.loc[:self.testdata, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        glm = sm.GLM(y, X, family=sm.families.Gaussian(link=sm.families.links.identity()))
        glm = glm.fit()
        self.logger.info(glm.summary())
        summaryfile = os.path.join(PLOTDIR, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(glm.summary().as_text())
        self.model = glm

    def plotResid(self):
        fig, axs = plt.subplots(nrows=2, ncols=1)
        resid = self.model.resid_response
        meanval = resid.mean()
        sd = resid.std()
        resid_std = (resid - meanval)/sd
        res = ss.kstest(resid_std, ss.norm.cdf)
        self.logger.info(res)

        xv = np.linspace(resid.min(), resid.max(), 100)
        yv = ss.norm.pdf(xv, meanval, sd)
        dates = self.df.loc[0:self.testdata, "DATE"].values
        axs[0].plot(dates, resid)
        axs[0].grid()
        axs[0].set_title("Residual Plot")
        axs[1].hist(resid, bins=40, density=True)
        axs[1].plot(xv, yv, lw=2)
        axs[1].grid()
        axs[1].set_title("Histogram of Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "trainResidNormal.jpeg"),
                    dpi=500)

    def plotTestResults(self, y, ypred):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        resid = y - ypred
        meanval = resid.mean()
        sd = resid.std()
        resid_std = (resid - meanval)/sd
        ksres = ss.kstest(resid_std, ss.norm.cdf)
        self.logger.info(ksres)
        xv = np.linspace(resid.min(), resid.max(), num=100)
        yv = ss.norm.pdf(xv, loc=meanval, scale=sd)
        dates = self.df.loc[self.testdata+1:, "DATE"].values
        axs[0].plot(dates, resid)
        axs[0].plot()
        axs[0].grid()
        axs[0].set_title("Residual Plot")
        axs[1].hist(resid, bins=40, density=True)
        axs[1].plot(xv, yv, lw=2)
        axs[1].grid()
        axs[1].set_title("Histogram of Residuals")
        axs[2].plot(dates, y, label="y")
        axs[2].plot(dates, ypred, "-.", label="ypred")
        axs[2].grid()
        axs[2].set_title("Predicted vs. Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, "testResidNormal.jpeg"),
                    dpi=500)

    def test(self):
        testdata = self.testdata + 1
        y = self.df.loc[testdata:, self.endog].values
        X = self.df.loc[testdata:, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        ypred = self.model.predict(X)
        self.plotTestResults(y, ypred)


if __name__ == "__main__":
    normal = NormalGLM("DGS10", ["DGS1MO", "DGS30", "SP500"])
    normal.fit()
    normal.plotResid()
    normal.test()
