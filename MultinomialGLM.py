import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import logging
import bisect
from sklearn.metrics import confusion_matrix
import seaborn as sns


DATADIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
PLOTDIR = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\plots"
logging.basicConfig(level=logging.DEBUG)


class MultinomialGLM(object):
    PERIOD = 5

    def __init__(self, endog, exog, trainPerc=0.9):
        self.logger = logging.getLogger(self.__class__.__name__)
        filename = os.path.join(DATADIR, endog + ".csv")
        self.trainPerc = trainPerc
        y = pd.read_csv(filename, parse_dates=["DATE"])
        self.convertColumnToFloat(y, endog)
        self.plotEndog(y, endog)
        self.beginIndex = 0
        self.retThresholds = [0.004]
        self.volatThresholds = [0.008]
        self.bucketNames = ["Low Ret Low Vol", "High Ret Low Vol",
                            "Low Ret High Vol", "High Ret High Vol"]
        for xi in exog:
            filename = os.path.join(DATADIR, xi + ".csv")
            x = pd.read_csv(filename, parse_dates=["DATE"])
            y = pd.merge(y, x, how="inner", on=["DATE"])
            self.convertColumnToFloat(y, xi)
        self.endIndex = y.shape[0] - self.PERIOD
        self.testdata = int(trainPerc * (self.endIndex - self.beginIndex)) + self.beginIndex
        y = self.calculateEndogExogVars(y, endog, exog)
        self.df = y
        self.model = None

    def getBucketNumber(self, retList, volatList):
        b1 = np.array([bisect.bisect_left(self.retThresholds, ret) for ret in retList])
        b2 = np.array([bisect.bisect_left(self.volatThresholds, volat) for volat in volatList])
        return (len(self.retThresholds) + 1) * b2 + b1

    def getBucketsAndName(self, num):
        b2, b1 = divmod(num, len(self.retThresholds) + 1)
        return b1, b2, self.bucketNames[num]

    def plotEndog(self, df, endog):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        y = df.loc[:, endog].values
        dates = df.DATE.values
        axs.plot(dates, y)
        axs.grid()
        axs.set_title(f"Closing Value for {endog}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTDIR, f"{endog}.jpeg"),
                    dpi=500)

    def calculateEndogExogVars(self, df, endog, exog):
        vals = df.loc[:, endog].values
        ret = vals[self.PERIOD:] / vals[0:-self.PERIOD] - 1
        volatility = self.volatility(ret, self.PERIOD)
        df.loc[:, "ret"] = 0
        df.loc[:, "volatility"] = 0
        df.loc[0:df.shape[0]-self.PERIOD-1, "ret"] = ret
        df.loc[0:df.shape[0]-self.PERIOD-1, "volatility"] = volatility
        self.logger.info(df.loc[0:self.testdata, ["ret", "volatility"]].describe())

        bucketNums = self.getBucketNumber(ret, volatility)
        df.loc[:, "bucket"] = 0
        df.loc[0:df.shape[0]-self.PERIOD-1, "bucket"] = bucketNums
        df.loc[:, "last5DayBucket"] = 0
        df.loc[self.PERIOD:, "last5DayBucket"] = bucketNums
        self.beginIndex = max(self.PERIOD, 11)
        self.endIndex = df.shape[0]-self.PERIOD
        self.testdata = int(self.trainPerc * (self.endIndex - self.beginIndex)) + self.beginIndex

        bucketNameList = ["Bucket_" + str(i) for i in range((len(self.retThresholds)+1)*(len(self.volatThresholds)+1))]
        last5DayBucketNameList = ["Last5Day_" + b for b in bucketNameList]
        for lnm, nm in zip(last5DayBucketNameList, bucketNameList):
            df.loc[:, nm] = 0.0
            df.loc[:, lnm] = 0.0

        for i in range(df.shape[0]-self.PERIOD):
            df.loc[i, "Bucket_%d"%bucketNums[i]] = 1.0
            df.loc[i+self.PERIOD, "Last5Day_Bucket_%d"%bucketNums[i]] = 1.0

        df.loc[:, "Last5DayRet"] = 0
        df.loc[self.PERIOD:, "Last5DayRet"] = ret
        df.loc[:, "Last5DayVolat"] = 0
        df.loc[self.PERIOD:, "Last5DayVolat"] = volatility

        ma5 = self.movingAverage(vals, 5)
        ma10 = self.movingAverage(vals, 10)
        df.loc[:, "ma5_10"] = 0.0
        df.loc[11:, "ma5_10"] = np.where(ma5[10:-1] > ma10[10:-1], 1.0, 0.0)

        ecols_list = []
        for exog1 in exog:
            evals = df.loc[:, exog1].values
            colnm = exog1 + "_diff"
            df.loc[:, colnm] = 0
            df.loc[self.PERIOD:, colnm] = evals[self.PERIOD:] - evals[0:-self.PERIOD]
            ecols_list.append(colnm)

        exog_cols = ["Last5DayRet", "Last5DayVolat", "ma5_10"] + ecols_list #last5DayBucketNameList + ["Last5DayRet", "Last5DayVolat", "ma5_10"]
        self.exog = exog_cols
        self.nvars = len(self.exog)
        self.endog = bucketNameList
        return df

    def volatility(self, arr, period):
        res = np.zeros(arr.shape[0], dtype=np.float64)
        sumval = np.sum(arr[0:period])
        sumsq = np.dot(arr[0:period], arr[0:period])
        for i in range(period, arr.shape[0]):
            res[i] = np.sqrt(sumsq/period - (sumval/period)**2)
            sumval += arr[i] - arr[i-period]
            sumsq += arr[i]*arr[i] - arr[i-period]*arr[i-period]
        return res

    def movingAverage(self, arr, period):
        res = np.zeros(arr.shape[0], dtype=np.float64)
        sumval = np.sum(arr[0:period])
        for i in range(period, arr.shape[0]):
            res[i] = sumval / period
            sumval += arr[i] - arr[i-period]
        return res

    def convertColumnToFloat(self, df, col):
        if (df.loc[:, col] == ".").sum() > 0:
            df.drop(np.where(df.loc[:, col] == ".")[0], inplace=True)
            df.reset_index(drop=True, inplace=True)
        df.loc[:, col] = df.loc[:, col].astype(np.float64)

    def fit(self):
        y = self.df.loc[self.beginIndex:self.testdata, self.endog].values
        X = self.df.loc[self.beginIndex:self.testdata, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        glm = sm.MNLogit(y, X)
        glm = glm.fit()
        glm.endog_names = self.endog
        self.logger.info(glm.summary(xname=['constant'] + self.exog))
        summaryfile = os.path.join(PLOTDIR, self.__class__.__name__ + ".txt")
        with open(summaryfile, 'w') as fh:
            fh.write(glm.summary(xname=['constant'] + self.exog).as_text())
        self.model = glm

    def plotTrainingConfusionMatrix(self):
        yendog = self.model.model.endog
        yhatv = self.model.predict(self.model.model.exog)
        mostProbableBucket = np.argmax(yhatv, axis=1)
        self.plotResults(yendog, mostProbableBucket, label="train")

    def plotResults(self, y_true, y_pred, label="test"):
        cm = confusion_matrix(y_true, y_pred)
        df = pd.DataFrame(cm.astype(np.int32), index=self.bucketNames, columns=self.bucketNames)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(df, annot=True, linewidths=0.5)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(os.path.join(PLOTDIR, f"confusionMatrix_{label}_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        self.logger.info(df)

    def test(self):
        testdata = self.testdata + 1
        X = self.df.loc[testdata:, self.exog].values
        X = sm.add_constant(X, has_constant="add")
        ypred = self.model.predict(X)
        mostProbableBucket = np.argmax(ypred, axis=1)
        actualBucket = np.argmax(self.df.loc[testdata:, self.endog].values, axis=1)
        self.plotResults(actualBucket, mostProbableBucket)


if __name__ == "__main__":
    glm = MultinomialGLM("SP500", ["DGS10", "DGS1MO"])
    # PAYEMS: Nonfarm payroll (monthly), seasonally adjusted
    # CCLACBW027SBOG: Loan on credit card and other revolving plans (weekly) 200
    # PCEPI: PCE price index (monthly) 100
    # DSPIC96: Real disposable income (monthly) 2000
    # MORTGAGE30US: 30 year mortgage rate (weekly) 100

    glm.fit()
    glm.plotTrainingConfusionMatrix()
    glm.test()
