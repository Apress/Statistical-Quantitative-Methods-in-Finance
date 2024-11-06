from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import statsmodels.api as sm
from scipy import stats


class Tobit1(GenericLikelihoodModel):
    def __init__(self, endog, exog, low=None, high=None, add_constant=True, method="bfgs"):
        super().__init__(endog=endog, exog=exog)
        assert (low is not None) or (high is not None), "both low and high cannot be None"
        if (low is not None) and (high is not None):
            assert low < high, "low must be strictly less than high"
        self.low = low
        self.high = high
        self.endog = endog
        self.exog = exog
        self.add_constant = add_constant
        if add_constant:
            self.exog = sm.add_constant(exog, has_constant="add", prepend=False)
        self.parameters = np.zeros(self.exog.shape[1] + 1, dtype=np.float64)
        self.parameters[-1] = 1.
        self.method = method

    def loglikeobs(self, params):
        error = self.endog - (self.exog @ params[:-1])
        ll = 0
        condition = np.ones(self.endog.shape[0], dtype=bool)
        if self.low is not None:
            ll += np.sum(stats.norm.logcdf(error[self.endog <= self.low], self.low, params[-1]))
            condition[self.endog <= self.low] = False
        if self.high is not None:
            ll += np.sum(stats.norm.logcdf(-error[self.endog > self.high], self.high, params[-1]))
            condition[self.endog > self.high] = False
        ll += np.sum(stats.norm.logpdf(error[condition], 0, params[-1]))
        return ll

    def fit(self, **kwargs):
        if "method" in kwargs:
            kwargs.pop("method")
        return super().fit(self.parameters, method=self.method, **kwargs)

    def predict(self, params, exog, *args, **kwargs):
        if self.add_constant:
            exog = sm.add_constant(exog, has_constant="add", prepend=False)
        pred = np.einsum("ij,j->i", exog, params[0:-1])
        if self.low is not None:
            pred = np.where(pred <= self.low, self.low, pred)
        if self.high is not None:
            pred = np.where(pred > self.high, self.high, pred)
        return pred

# unittest below

import unittest
import logging
import statsmodels.api as sm

logging.basicConfig(level=logging.DEBUG)


class TobitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        x = np.arange(-5, 5, 0.01)
        np.random.seed(1024)
        coeff = 3
        variance = 1
        l1 = 0.0
        sdev = np.sqrt(variance)
        epsilon = sdev * np.random.standard_normal(x.shape[0])
        ystar = 1 + coeff * x + epsilon
        y = np.where(ystar < l1, l1, ystar)
        self.x = x
        self.y = y
        self.l1 = l1
        self.tobit = None
        self.trainTestRatio = 0.9

    def test_regression(self):
        training = int(self.trainTestRatio * self.x.shape[0])
        x = self.x[0:training, np.newaxis]
        y = self.y[0:training]
        self.tobit = Tobit1(y, x, low=self.l1, add_constant=True)

        res = self.tobit.fit()
        self.logger.info(res.summary())
        self.assertIsNotNone(res)

        exogTest = self.x[training:]
        testPred = self.tobit.predict(res.params, exogTest)
        actual = self.y[training:]
        diff = testPred - actual
        mse1 = np.sqrt(np.mean(diff * diff))

        # fit OLS model
        x = sm.add_constant(self.x[0:training, np.newaxis], has_constant="add")
        olsModel = sm.OLS(self.y[0:training], x).fit()
        testX = sm.add_constant(self.x[training:, np.newaxis], has_constant="add")
        olsPred = olsModel.predict(testX)
        self.logger.info(olsModel.summary())
        diff = olsPred - actual
        mse2 = np.sqrt(np.mean(diff * diff))
        self.logger.info("RMSE from tobit: %f, from OLS: %f", mse1, mse2)
        self.assertLess(mse1, mse2)
