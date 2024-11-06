import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.DEBUG)


class BankingRF(object):
    """ Predict if clients subscribes to a banking product (term deposit) using random forest """
    LOGGER = logging.getLogger("BankingRF")

    def __init__(self, datadir: str, filename: str = "bank-additional-full.csv", testing: float = 0.1,
                 ntree: int = 100, fill_na_features=True, oob_score_convergence=False) -> None:
        """
        :param datadir: Data directory containing input file
        :param filename: File name containing the data
        :oaram testing: percentage of data to set aside as testing data
        :param ntree: Number of decision trees in random forest
        :param fill_na_features: Fill NA categorical features with default value
        :param oob_score_convergence: Plot OOB score convergence
        :rtype: None
        """
        df = pd.read_csv(os.path.join(datadir, filename), sep=";")
        excludeCols = ["duration"]
        df.drop(columns=excludeCols, inplace=True)
        self.resultCol = "y"
        df.loc[:, self.resultCol] = df.loc[:, self.resultCol].map({'yes': True, 'no': False})
        self.df = df

        nrows = self.df.shape[0]
        training = 1.0 - testing
        self.trainDf = self.df.loc[0:int(training * nrows), :].reset_index(drop=True)
        self.testDf = self.df.loc[int(training * nrows):, :].reset_index(drop=True)
        features = list(self.trainDf.columns)
        features.remove(self.resultCol)
        self.featureNames = features
        self.numericCols = []
        self.normalizeCols = {}
        self.categoricalCols = []
        self.categoricalMap = {}
        self._processColumns()
        self.fillNa = fill_na_features
        self.oobScore = oob_score_convergence
        self.model = RandomForestClassifier(n_estimators=ntree, random_state=0)
        self.trainModel()

    def _processCategoricalCols(self, df: pd.DataFrame) -> None:
        """
        Process categorical columns by creating a mapping
        :param df: training dataframe
        :rtype: None
        """
        for col in self.categoricalCols:
            unique = np.sort(df.loc[:, col].unique())
            self.categoricalMap[col] = {u: i for i, u in enumerate(unique)}

    def _applyCategoricalMapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mappping to convert categorical columns to integers
        :rtype: pd.DataFrame with mapped categorical columns
        """
        for col in self.categoricalCols:
            df.loc[:, col] = df.loc[:, col].map(self.categoricalMap[col])
            return df

    def _normalizeNumericCols(self, trainingDf: pd.DataFrame) -> None:
        """
        Calclate normalizing params for numeric columns
        :param trainingDf:
        :return: None
        """
        for col in self.numericCols:
            mean = trainingDf.loc[:, col].mean()
            sd = trainingDf.loc[:, col].std()
            self.normalizeCols[col] = (mean, 2 * sd)

    def _applyNormalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization as col = (x-mean)/(2*sd)
        :param df:
        :return: df
        """
        for col in self.numericCols:
            mean, sd2 = self.normalizeCols[col]
            df.loc[:, col] = (df.loc[:, col].values - mean) / sd2
        return df

    def _processColumns(self) -> None:
        """
        Process input columns from dataframe. Dataframe is in self.df
        :return: None
        """
        df = self.trainDf

        # identify categorical columns
        cols = list(df.columns)
        cols.remove(self.resultCol)
        for col in cols:
            if df.dtypes[col].name == "object":
                self.categoricalCols.append(col)
            else:
                self.numericCols.append(col)

        self._processCategoricalCols(df)
        self.trainDf = self._applyCategoricalMapping(self.trainDf)
        self.testDf = self._applyCategoricalMapping(self.testDf)

        self._normalizeNumericCols(df)
        self.trainDf = self._applyNormalization(self.trainDf)
        self.testDf = self._applyNormalization(self.testDf)

    def _plotOOBError(self):
        estimators = range(15, 150)
        X = self.trainDf.loc[:, self.featureNames].values
        y = self.trainDf.loc[:, self.resultCol].values
        err = []
        for nest in estimators:
            rf = RandomForestClassifier(n_estimators=nest, oob_score=True, random_state=0)
            rf.fit(X, y)
            err.append(1 - rf.oob_score_)
            errdf = pd.DataFrame({"Number of Trees": list(estimators), "OOB Error Rate": err})
        sns.lineplot(data=errdf, x="Number of Trees", y="OOB Error Rate")
        plt.show()

    def _plotConfusionMatrix(self, labels: np.ndarray, predictions: np.ndarray) -> None:
        cm = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", linewidths=0.25, ax=ax)
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def _calcMeasures(self) -> None:
        """ Calculate and plot measures after fitting random forest """
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        shortFeatName = []
        for feature in self.featureNames:
            if len(feature) > 8:
                feature = feature[0:2] + ".." + feature[-3:]
            shortFeatName.append(feature)
            impdf = pd.DataFrame({"Feature": shortFeatName,
                                  "GINI Importance": importances,
                                  "sd": std})

        ax = sns.barplot(data=impdf, x="Feature", y="GINI Importance")
        ax.errorbar(data=impdf, x="Feature", y="GINI Importance", ls='', lw='3', color="black")
        plt.show()

        self._plotOOBError()

    def trainModel(self) -> None:
        """
        Train the random forest classifier on training dataset
        :return:
        """
        X = self.trainDf.loc[:, self.featureNames].values
        y = self.trainDf.loc[:, self.resultCol].values
        self.model.fit(X, y)
        if self.oobScore:
            self._calcMeasures()

        ypred = self.model.predict(X)
        self._plotConfusionMatrix(y, ypred)

        Xtest = self.testDf.loc[:, self.featureNames].values
        ytest = self.testDf.loc[:, self.resultCol].values

        rowsWithNan = np.where(np.isnan(Xtest).sum(axis=1))[0]
        if rowsWithNan.shape[0]:
            self.LOGGER.info("Some categorical variables in test data were not present in training!")
            if self.fillNa:
                Xtest = np.nan_to_num(Xtest)  # fill missing categorical variables with 0
            else:
                rowsWithoutNan = np.array([i for i in range(Xtest.shape[0]) if i not in set(rowsWithNan)])
                Xtest = Xtest[rowsWithoutNan, :]
                ytest = ytest[rowsWithoutNan]

        testPred = self.model.predict(Xtest)

        self._plotConfusionMatrix(ytest, testPred)

    def testModel(self, testDf: pd.DataFrame) -> None:
        """
        Test the model using provided testing data
        :param testDf:
        :return: None
        """
        if testDf is None:
            testDf = self.testDf

        X = testDf.loc[:, self.featureNames].values
        pred = self.model.predict(X)
        y = testDf.loc[:, self.resultCol].values
        self._plotConfusionMatrix(y, pred)


if __name__ == "__main__":
    rf = BankingRF(r"C:\prog\cygwin\home\samit_000\RLPy\data\book\bank-additional", ntree=110)

