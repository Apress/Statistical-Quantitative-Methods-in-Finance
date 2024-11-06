import numpy as np
import logging
import enum
from scipy.stats import norm
from typing import Tuple


@enum.unique
class TobitType(enum.IntEnum):
    """ Type of Tobit Regression Model """
    TYPE1 = 1,
    TYPE2 = 2,
    TYPE3 = 3


class RegressionResult(object):
    """ Regression results class """
    def __init__(self):
        self._history = []
        self._params = None
        self._vars = None
        self._fittedModel = None

    @property
    def history(self):
        return self._history

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, value):
        self._params = value

    @property
    def varnames(self):
        return self._vars

    @varnames.setter
    def varnames(self, value):
        self._vars = value

    def setModel(self, fitted_model):
        self._fittedModel = fitted_model

    def appendDiff(self, diff):
        """
        Append parameter diff across iterations to history
        :param diff: mean sum of square diff in model parameters across iterations
        """
        self._history.append(diff)

    def predict(self, exog: np.ndarray) -> np.ndarray:
        """
        Predict the model output. Model must have been fitted to data
        :param exog: exogeneous variables (X)
        :return: model output (y)
        """
        return self._fittedModel._predict(exog, self._params)


class TobitRegression(object):
    """ Tobit (censored) regression model """
    DEFAULT_DTYPE = np.float32

    def __init__(self, low=None, high=None, type=TobitType.TYPE1, diff_thresh=1E-10, niters=1000, dtype=np.float32):
        """
        Initialize the regression model
        :param low: Low threshold for censoring
        :param high: High threshold for censoring
        :param type: Tobit type 1, 2 or 3. Currently only type 1 is supported. Future releases will add support for
        type 2 and type 3 Tobit regressions
        :param diff_thresh: Stop iterations when diff between successive iterations becomes lower than this threshold
        :param niters: Maximum number of iterations
        :param dtype: data type of numeric variables. Default is 32 bit float
        """
        assert (low is not None) or (high is not None), "both low and high cannot be None"
        if (low is not None) and (high is not None):
            assert low < high, "low must be strictly less than high"
        self.DEFAULT_DTYPE = dtype
        self.diffThreshold = diff_thresh
        self.maxIters = niters
        self.delta = None
        self.gamma = None
        self.low = low
        self.high = high
        self.aIndex = np.array([])
        self.bIndex = np.array([])
        self.midIndex = np.array([])
        self.funcVal = None
        self.funcDeriv = None
        self.categoricalMap = {}
        self.colNames = []
        self.categoricalColInd = None
        self.hasConst = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def parameters(self):
        return self.delta/self.gamma, 1.0/self.gamma

    def initializeParams(self, nexog: int, endog: np.ndarray) -> None:
        """
        Initialize model parameters
        :param nexog: number of exogeneous (independent) variables
        :param endog: endogeneous (dependent) variable array
        """
        self.delta = np.ones(nexog, dtype=self.DEFAULT_DTYPE)
        self.funcVal = np.zeros(nexog + 1, dtype=self.DEFAULT_DTYPE)
        self.funcDeriv = np.zeros((nexog + 1, nexog + 1), dtype=self.DEFAULT_DTYPE)
        self.gamma = 1.0
        indicA = 0
        indicB = 0
        if self.low is not None:
            indicA = np.less_equal(endog, self.low)
            if indicA.sum():
                self.aIndex = np.where(indicA)[0]
        if self.high is not None:
            indicB = np.greater_equal(endog, self.high)
            if indicB.sum():
                self.bIndex = np.where(indicB)[0]
        self.midIndex = np.where(1 - indicA - indicB)[0]

    def lowLikelihood(self, exog: np.ndarray) -> float:
        """
        Calculate log likelihood of output value at lower threshold
        :param exog: Array of exogeneous variables
        :return: Log likelihood of output at lower threshold
        """
        required = self.aIndex
        xdelta = self.low * self.gamma - np.einsum("ij,j->i", exog[required, :], self.delta)
        cdf = norm.cdf(xdelta)
        return np.log(cdf).sum()

    def highLikelihood(self, exog: np.ndarray) -> float:
        """
        Calculate log likelihood of output value at higher threshold
        :param exog: Array of exogeneous variables
        :return: Log likelihood of output value at lower threshold
        """
        required = self.bIndex
        xdelta = -self.high * self.gamma + np.einsum("ij,j->i", exog[required, :], self.delta)
        cdf = norm.cdf(xdelta)
        return np.log(cdf).sum()

    def midLikelihood(self, exog: np.ndarray, endog: np.ndarray) -> float:
        """
        Calculate log likelihood of output value not hitting lower or higher thresholds (in uncensored region)
        :param exog: Array of exogeneous variables
        :param endog: Array of endogeneous (output) variables
        :return: log likelihood of output value in uncensored region
        """
        required = self.midIndex
        xdelta = self.gamma * endog[required] - np.einsum("ij,j->i", exog[required, :], self.delta)
        pdf = norm.pdf(xdelta)
        return np.log(pdf * self.gamma).sum()

    def logLikelihood(self, exog: np.ndarray, endog: np.ndarray) -> float:
        """
        Log likelihood of data subject to lower and higher thresholds (censoring)
        :param exog: Array of exogeneous variables
        :param endog: Array of endogeneous variables
        :return: Log likelihood of data subject to lower and higher thresholds
        """
        result = 0
        if self.low is not None:
            result += self.lowLikelihood(exog)
        if self.high is not None:
            result += self.highLikelihood(exog)

        result += self.midLikelihood(exog, endog)
        return result

    def lowFunc(self, exog: np.ndarray, val: np.ndarray) -> None:
        """
        Function (= derivative of log likelihood) at lower censoring threshold
        :param exog: array of exogeneous variables
        :param val: function value at lower threshold
        """
        required = self.aIndex
        xdelta = self.low * self.gamma - np.einsum("ij,j->i", exog[required, :], self.delta)
        den = norm.cdf(xdelta)
        num = norm.pdf(xdelta)
        limindex = (den == 0.) & (num == 0.)
        den = np.where(limindex, 1.0, den)
        num = np.where(limindex, 1.0, num)
        frac = num / den
        val[0:-1] += -np.einsum("i,ij->j", frac, exog[required, :])
        val[-1] += frac.sum() * self.low

    def highFunc(self, exog: np.ndarray, val: np.ndarray) -> None:
        """
        Function at higher censoring threshold
        :param exog: array of exogeneous variables
        :param val: value of function at upper censoring threshold
        """
        required = self.bIndex
        xdelta = -self.high * self.gamma + np.einsum("ij,j->i", exog[required, :], self.delta)
        den = norm.cdf(xdelta)
        num = norm.pdf(xdelta)
        limindex = (den == 0.) & (num == 0.)
        den = np.where(limindex, 1.0, den)
        num = np.where(limindex, 1.0, num)
        frac = num / den
        val[0:-1] += np.einsum("i,ij->j", frac, exog[required, :])
        val[-1] += -frac.sum() * self.high

    def midFunc(self, exog: np.ndarray, endog: np.ndarray, val: np.ndarray) -> None:
        """
        Function value in uncensored region
        :param exog: array of exogeneous variables
        :param endog: array of output variables
        :param val: value of function
        """
        required = self.midIndex
        xdelta = self.gamma * endog[required] - np.einsum("ij,j->i", exog[required, :], self.delta)
        val[0:-1] += np.einsum("i,ij->j", xdelta, exog[required, :])
        val[-1] += 1.0 / self.gamma - np.multiply(xdelta, endog[required]).sum()

    def function(self, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """
        Function (derivative of log likelihood) that we are trying to determine a root of
        :param exog: array of exogeneous variables (X)
        :param endog: array of output variables (y)
        :return: array of function values evaluated at specified X and y
        """
        val = self.funcVal
        val[:] = 0
        if self.aIndex.shape[0]:
            self.lowFunc(exog, val)
        if self.bIndex.shape[0]:
            self.highFunc(exog, val)
        self.midFunc(exog, endog, val)
        return val

    def lowDeriv(self, exog: np.ndarray, val: np.ndarray) -> None:
        """
        Derivative of function (which was derivative of log likelihood) at lower censoring threshold
        :param exog: array of exogeneous variables
        :param val: Derivative of function (2 dimensional array) at lower censoring threshold
        """
        required = self.aIndex
        xdelta = self.low * self.gamma - np.einsum("ij,j->i", exog[required, :], self.delta)
        den = norm.cdf(xdelta)
        num = norm.pdf(xdelta)
        limindex = (den == 0.) & (num == 0.)
        den = np.where(limindex, 1.0, den)
        num = np.where(limindex, 1.0, num)
        frac = num / den
        termfrac = np.multiply(frac, 1 - frac)
        prod = np.multiply(frac, xdelta)
        termprod = np.multiply(prod, 1 - prod)
        val[0:-1, 0:-1] += np.einsum("i,ij,ik->jk", termprod, exog[required, :], exog[required, :])
        val[-1, -1] += termfrac.sum() * self.low * self.low
        crossDeriv = -self.low * np.einsum("i,ij->j", termfrac, exog[required, :])
        val[-1, 0:-1] += crossDeriv
        val[0:-1, -1] += crossDeriv

    def highDeriv(self, exog: np.ndarray, val: np.ndarray) -> None:
        """
         Derivative of function (which was derivative of log likelihood) at upper censoring threshold
        :param exog: array of exogeneous variables
        :param val: Derivative of function (2 dimensional array) at uppper censoring threshold
        """
        required = self.bIndex
        xdelta = -self.high * self.gamma + np.einsum("ij,j->i", exog[required, :], self.delta)
        den = norm.cdf(xdelta)
        num = norm.pdf(xdelta)
        limindex = (den == 0.) & (num == 0.)
        den = np.where(limindex, 1.0, den)
        num = np.where(limindex, 1.0, num)
        frac = num / den
        termfrac = np.multiply(frac, 1 - frac)
        prod = np.multiply(frac, xdelta)
        termprod = np.multiply(prod, 1 - prod)
        val[0:-1, 0:-1] += np.einsum("i,ij,ik->jk", termprod, exog[required, :], exog[required, :])
        val[-1, -1] += termfrac.sum() * self.high * self.high
        crossDeriv = -self.high * np.einsum("i,ij->j", termfrac, exog[required, :])
        val[-1, 0:-1] += crossDeriv
        val[0:-1, -1] += crossDeriv

    def midDeriv(self, exog: np.ndarray, endog: np.ndarray, val: np.ndarray) -> None:
        """
        Derivative of function (which was derivative of log likelihood) in uncensored region
        :param exog: array of exogeneous variables
        :param endog: array of output variables
        :param val: Derivative of function (2 dimensional array) in uncensoring region
        """
        required = self.midIndex
        val[0:-1, 0:-1] += -np.einsum("ij,ik->jk", exog[required, :], exog[required, :])
        val[-1, -1] += -(1.0 / (self.gamma * self.gamma) +
                         np.multiply(endog[required], endog[required]).sum())
        yixij = np.einsum("i,ij->j", endog[required], exog[required, :])
        val[-1, 0:-1] += yixij
        val[0:-1, -1] += yixij

    def derivative(self, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """
        Derivative of function (which was derivative of log likelihood function)
        :param exog: array of exogeneous variables
        :param endog: array of output variables
        :return: Derivative (2 dimensional array) of function. Function is the derivative of log likelihood
        """
        deriv = self.funcDeriv
        deriv[:, :] = 0
        if self.aIndex.shape[0]:
            self.lowDeriv(exog, deriv)
        if self.bIndex.shape[0]:
            self.highDeriv(exog, deriv)
        self.midDeriv(exog, endog, deriv)
        return deriv

    def newtonStep(self, exog: np.ndarray, endog: np.ndarray) -> float:
        """
        Perform a Newton step to update model parameters. \delta x = - f(x) / gradient(f(x))
        :param exog: array of exogeneous variables
        :param endog: array of output variables
        :return: diff or the change in parameters
        """
        fx = self.function(exog, endog)
        jacobian = self.derivative(exog, endog)
        step = np.linalg.solve(jacobian, -fx)
        self.delta += step[0:-1]
        self.gamma += step[-1]
        self.gamma = abs(self.gamma)
        return np.multiply(step, step).sum() / step.shape[0]

    def coerceData(self, exog: np.ndarray, endog: np.ndarray, categorical: tuple,
                   col_names: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Coerce data to default datatype
        :param exog:
        :param endog:
        :param categorical:
        :param col_names:
        :return:
        """
        self.categoricalMap = {}
        catset = set(categorical)
        noncat = [i for i in range(exog.shape[1]) if i not in catset]
        colnames = [col_names[i] for i in range(exog.shape[1]) if i not in catset]
        concatarr = [exog[:, noncat]]
        for cat in categorical:
            distinct = sorted(list(set(exog[:, cat])))
            self.categoricalMap[cat] = distinct
            catcols = np.zeros((exog.shape[0], len(distinct)-1), dtype=self.DEFAULT_DTYPE)
            for i, d in enumerate(distinct[0:-1]):
                catcols[:, i] = np.where(exog[:, cat] == d, 1, 0)
            concatarr.append(catcols)
            colnames.extend(["%s_%s" % (col_names[cat], si) for si in distinct[0:-1]])
        modExog = np.concatenate(concatarr, axis=1)
        self.colNames = colnames
        return modExog.astype(self.DEFAULT_DTYPE), endog.astype(self.DEFAULT_DTYPE)

    def fit(self, endog: np.ndarray, exog: np.ndarray, include_constant: bool = True, categorical: tuple = (),
            col_names: tuple = ()) -> RegressionResult:
        """
        Fit the Tobit regression model to the data
        :param endog: array of output variables
        :param exog: array of exogeneous variables
        :param include_constant: Add constant to exogeneous variables array
        :param categorical: tuple containing indices of categorical columns
        :param col_names: Column names (for ease of identification of fitted parameters)
        :return: Regression results
        """
        if not col_names:
            col_names = tuple("col_%d" % i for i in range(exog.shape[1]))
        self.hasConst = include_constant
        if include_constant:
            const = np.ones((exog.shape[0], 1), dtype=self.DEFAULT_DTYPE)
            exog = np.concatenate((exog, const), axis=1)
            col_names = col_names + ("intercept",)
        self.categoricalColInd = categorical
        exog, endog = self.coerceData(exog, endog, categorical, col_names)
        self.initializeParams(exog.shape[1], endog)
        diff = self.diffThreshold + 1
        iters = 0
        res = RegressionResult()
        while (diff > self.diffThreshold) and (iters < self.maxIters):
            diff = self.newtonStep(exog, endog)
            iters += 1
            res.appendDiff(diff)
        self.logger.info("Iterations: %d, final residual: %f", iters, diff)
        res.parameters = np.array([self.delta, self.gamma], dtype=self.DEFAULT_DTYPE)
        res.varnames = self.colNames
        return res

    def _coerceExogForPred(self, exog: np.ndarray) -> np.ndarray:
        """
        Coerce exogeneous variables array during prediction to handle categorical variables
        :param exog: array of exogeneous variables
        :return: Processed exogeneous variables with categorical columns converted to indicator variable columns
        """
        catset = set(self.categoricalColInd)
        noncat = [i for i in range(exog.shape[1]) if i not in catset]
        concatarr = [exog[:, noncat]]
        for cat in self.categoricalColInd:
            distinct = sorted(list(set(exog[:, cat])))
            catcols = np.zeros((exog.shape[0], len(distinct) - 1), dtype=self.DEFAULT_DTYPE)
            for i, d in enumerate(distinct[0:-1]):
                catcols[:, i] = np.where(exog[:, cat] == d, 1, 0)
            concatarr.append(catcols)
        modExog = np.concatenate(concatarr, axis=1)
        return modExog.astype(self.DEFAULT_DTYPE)

    def _inverseMillsRatio(self, vals):
        return norm.pdf(vals) / norm.cdf(vals)

    def _predict(self, exog: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Predict the output of the model using exogeneous variables as input. This method should not be called directly.
        Use predict method from RegressionResults object
        :param exog: exogeneous variables (X)
        :param params: fitted model parameters (provided by RegressionResult object)
        :return: output value from the model (y)
        """
        if self.hasConst:
            exog2 = np.ndarray((exog.shape[0], exog.shape[1] + 1), dtype=self.DEFAULT_DTYPE)
            exog2[:, 0:-1] = exog
            exog2[:, -1] = 1.0
            exog = exog2

        exog = self._coerceExogForPred(exog)
        vals = np.einsum("ij,j->i", exog, params[0]/params[1])
        vals = vals + 1.0/params[1] * self._inverseMillsRatio(vals)
        if self.low:
            vals = np.where(vals <= self.low, self.low, vals)
        if self.high:
            vals = np.where(vals > self.high, self.high, vals)
        return vals