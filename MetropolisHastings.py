import numpy as np
import pandas as pd
import os
import logging
from abc import ABC, abstractmethod
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


class MetropolisHastings(ABC):
    def __init__(self, burnIn=1000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.burnIn = burnIn

    @abstractmethod
    def sampleFromProposalDensity(self, state0):
        raise NotImplementedError("Base class needs to implement")

    @abstractmethod
    def proposalDensity(self, state0, state1):
        raise NotImplementedError("Base class needs to implement")

    @abstractmethod
    def targetProb(self, state, params):
        raise NotImplementedError("Base class needs to implement")

    def sample(self, N, initial, params, burnIn=None):
        if burnIn is None:
            burnIn = self.burnIn
        samples = np.zeros(N, dtype=np.float64)
        state0 = initial
        i = 0
        while i < burnIn + N:
            state = self.sampleFromProposalDensity(state0)
            fac1 = self.proposalDensity(state, state0) / self.proposalDensity(state0, state)
            fac2 = self.targetProb(state, params) / self.targetProb(state0, params)
            acceptanceProb = min(fac1 * fac2, 1)
            u = np.random.random(1)
            if u <= acceptanceProb:
                state0 = state
                if i >= burnIn:
                    samples[i - burnIn] = state0
                i += 1

        return samples


class Garch11Model(GenericLikelihoodModel):
    def __init__(self, endog, exog):
        super().__init__(endog=endog, exog=exog)
        self.endog = endog
        self.exog = sm.add_constant(exog, has_constant="add")
        assert self.exog.shape[1] == 3
        self.parameters = np.random.random(3)

    def loglikeobs(self, params):
        pred = np.einsum("ij,j->i", self.exog, params)
        return np.sum(stats.norm.logpdf(pred, self.endog, 1))

    def fit(self, **kwargs):
        return super().fit(self.parameters, method="bfgs")

    def predict(self, exog):
        exog = sm.add_constant(exog, has_constant="add")
        return np.einsum("ij,j->i", exog, self.parameters)


class SP500ReturnPosterior(MetropolisHastings):
    PRICE_COL = "Close"
    PERIOD = 5

    def __init__(self, dirname, security, trainTestRatio=0.9):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dirname = dirname
        self.df = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        self.trainTestRatio = trainTestRatio
        self.ntraining = int(self.df.shape[0] * trainTestRatio)
        self.garchModel = None
        self.calculateEndogExogVars()
        self.volatForProb = None

    def calculateEndogExogVars(self):
        price = self.df.loc[:, self.PRICE_COL].values
        returns = price[1:] / price[0:-1] - 1
        self.df.loc[:, "returns"] = 0
        self.df.loc[1:, "returns"] = returns
        self.df.loc[:, "returns_square"] = self.df.loc[:, "returns"] ** 2
        self.df.loc[:, "volat"] = 0
        self.df.loc[:, "lagged_volat"] = 0
        sumsq = np.sum(returns[0:self.PERIOD] ** 2)
        for i in range(self.PERIOD, self.df.shape[0]-1, 1):
            self.df.loc[i, "volat"] = sumsq / self.PERIOD
            self.df.loc[i+1, "lagged_volat"] = self.df.loc[i, "volat"]
            sumsq += returns[i] * returns[i] - returns[i - self.PERIOD] * returns[i - self.PERIOD]

    def fitGarch(self):
        endog = self.df.loc[self.PERIOD+1:self.ntraining, "volat"].values
        exog = self.df.loc[self.PERIOD+1:self.ntraining, ["returns_square", "lagged_volat"]].values

        self.garchModel = Garch11Model(endog=endog, exog=exog)
        res = self.garchModel.fit()
        self.logger.info(res.summary())
        self.garchModel.parameters = res.params
        self.volatForProb = res.params[0] / (1 - res.params[2])

    def fitMHSampler(self):
        state = self.df.loc[self.ntraining, "returns"]
        mu = np.mean(self.df.loc[self.ntraining - self.PERIOD:self.ntraining, "returns"].values)
        volat = self.df.loc[self.ntraining, "volat"]
        params = (mu, volat)
        self.sample(1, state, params)

    def fit(self):
        self.fitGarch()
        self.fitMHSampler()

    def sampleFromProposalDensity(self, state0):
        return np.random.normal(size=1, loc=state0, scale=self.volatForProb)

    def proposalDensity(self, state0, state1):
        return stats.norm.pdf(state0 - state1, 0, 1)

    def targetProb(self, state, params):
        mu, volat = params
        nu = 2 * volat / (1 - volat)
        return (1 + (state - mu)**2/nu) ** (-(nu+1)/2) * stats.norm.pdf(state, mu, 1)

    def test(self):
        exog = self.df.loc[self.ntraining:, ["returns_square", "lagged_volat"]].values
        actual = self.df.loc[self.ntraining:, "volat"].values
        predictedVol = self.garchModel.predict(exog)
        sampledVol = np.zeros(self.df.shape[0]-1-self.ntraining, dtype=np.float64)
        x = self.df.loc[self.ntraining:, "Date"].values

        for i in range(self.ntraining, self.df.shape[0]-1, 1):
            vol = predictedVol[i-self.ntraining]
            self.volatForProb = self.df.loc[i, "lagged_volat"]
            mu = np.mean(self.df.loc[i-self.PERIOD:i, "returns"].values)
            initial = self.df.loc[i, "returns"]
            params = (mu, vol)
            returns = self.sample(20, initial, params, burnIn=0)
            sampledVol[i-self.ntraining] = np.std(returns)

        plt.figure(figsize=(10, 10))
        plt.plot(x[0:-1], sampledVol, label="Sampled")
        plt.plot(x[0:-1], predictedVol[0:-1], label="GARCH(1,1)")
        plt.plot(x[0:-1], actual[0:-1], label="Empirical")
        plt.grid()
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Daily Volatility")
        plt.savefig(os.path.join(self.dirname, "mcmc_variance.jpeg"),
                    dpi=500)
        plt.show()


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    posterior = SP500ReturnPosterior(dirname, "SPY")
    np.random.seed(32)
    posterior.fit()
    posterior.test()