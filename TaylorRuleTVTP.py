import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import os
import matplotlib.pyplot as plt
from src.TaylorRule import TaylorRule

logging.basicConfig(level=logging.DEBUG)


class TaylorRuleTVTP(TaylorRule):
    def fitRegimeSwitch(self):
        y, X = self.trainData()
        ntrain = int(self.df.shape[0] * self.trainTestSplit)
        date = self.df.loc[0:ntrain, "DATE"].values
        np.random.seed(1024)
        XWithConst = sm.add_constant(X, has_constant="add")
        self.markovModel = sm.tsa.MarkovRegression(endog=y, k_regimes=3, trend='c', exog=X,
                                                   exog_tvtp=XWithConst,
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

        fig, axes = plt.subplots(3, figsize=(10, 7))

        ax = axes[0]
        ax.plot(date, y + self.df.loc[0:ntrain, "pi"].values)
        ax.set(title="EFFR")
        ax.grid()

        ax = axes[1]
        ax.plot(date, X[:, 0])
        ax.set(title="Inflation Above Target (2%)")
        ax.grid()

        ax = axes[2]
        ax.plot(date, X[:, 1])
        ax.set(title="Output Gap")
        ax.grid()

        fig.tight_layout()
        plt.savefig(os.path.join(self.dirname, f"regime_vars_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    trule = TaylorRuleTVTP(dirname, "DFF", "A191RI1Q225SBEA", "fredgraph_OutputGap", trainTestSplit=1.0)
    trule.fit()
