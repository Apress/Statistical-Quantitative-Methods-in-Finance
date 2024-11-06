import pandas as pd
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
import logging
import os

logging.basicConfig(level=logging.DEBUG)


class CEVModel(GMM):
    PRICE = "Close"
    DELTAT = 1.0/251.0
    logger = logging.getLogger("CEVModel")
    N_MOMS = 4
    N_PARAMS = 3

    @staticmethod
    def calculateInstrumentalVars(dirname, security):
        CEVModel.logger = logging.getLogger("CEVModel")
        df = pd.read_csv(os.path.join(dirname, f"{security}.csv"), parse_dates=["Date"])
        prices = df.loc[:, CEVModel.PRICE].values
        exog = np.column_stack((prices[0:-2], prices[1:-1], prices[2:])) # (S(t-1), S(t), S(t+1))
        endog = np.zeros(exog.shape[0], dtype=np.float32) # dummy
        const = np.ones(exog.shape[0], dtype=np.int8)
        instruments = np.column_stack((const, prices[0:-2], prices[1:-1])) # (1, S(t-1), S(t))
        return endog, exog, instruments

    def momcond(self, params):
        mu, sigma, gamma = params
        x = self.exog
        z = self.instrument
        gtheta = x[:, 2] - x[:, 1] - mu * x[:, 1] * CEVModel.DELTAT
        moment = np.multiply(z, gtheta[:, np.newaxis])
        logReturn = np.log(x[:, 2] / x[:, 1])
        meanLogRet = np.mean(logReturn)
        volat = (sigma ** 2) * (x[:, 1] ** (2*(gamma - 1))) * CEVModel.DELTAT
        fourthMoment = ((logReturn - meanLogRet) ** 2) - volat
        moment = np.column_stack((moment, fourthMoment))
        return moment

    def fitCEV(self):
        params = np.array([0, 0.0001, 0.7])
        result = super().fit(params, maxiter=100, optim_method='bfgs', weights_method='hac', wargs=dict(centered=False, maxlag=1))
        CEVModel.logger.info(result.summary(xname=['mu', 'sigma', 'gamma']))
        


if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    security = "SPY"
    endog, exog, instruments = CEVModel.calculateInstrumentalVars(dirname, security)
    model1 = CEVModel(endog, exog, instruments, k_moms=CEVModel.N_MOMS, k_params=CEVModel.N_PARAMS)
    model1.fitCEV()