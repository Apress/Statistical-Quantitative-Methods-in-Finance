import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.pyplot import cm

logging.basicConfig(level=logging.DEBUG)


class KMeansCluster:
    PERIOD = 5
    VOLAT_PERIOD = 21

    def __init__(self, dirname, spy, K=4):
        self.logger = logging.getLogger(self.__class__.__name__)
        filename = os.path.join(dirname, f"{spy}.csv")
        self.df = pd.read_csv(filename, parse_dates=["Date"])
        self.priceCol = "Close"
        self.dirname = dirname
        self.calculateVolatAndReturns()
        self.model = KMeans(n_clusters=K)

    def calculateVolatAndReturns(self):
        prices = self.df.loc[:, self.priceCol].values
        returns1Day = prices[1:]/prices[0:-1] - 1
        returnPeriod = prices[self.PERIOD:]/prices[0:-self.PERIOD] - 1

        self.df.loc[:, "volat"] = 0
        for i in range(self.VOLAT_PERIOD, self.df.shape[0], 1):
            self.df.loc[i, "volat"] = np.std(returns1Day[i-self.VOLAT_PERIOD:i])

        self.df.loc[:, "return"] = 0
        self.df.loc[0:self.df.shape[0]-self.PERIOD-1, "return"] = returnPeriod

        x = self.df.loc[self.VOLAT_PERIOD:self.df.shape[0]-self.PERIOD-1, "volat"].values
        y = self.df.loc[self.VOLAT_PERIOD:self.df.shape[0]-self.PERIOD-1, "return"].values
        plt.scatter(x, y)
        plt.grid()
        plt.xlabel("Volatility")
        plt.ylabel("5-Day Return")
        plt.title("Scatterplot of Points")
        plt.savefig(os.path.join(self.dirname, f"scatterplot_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()

    def fit(self):
        x = self.df.loc[self.VOLAT_PERIOD:self.df.shape[0] - self.PERIOD - 1, "volat"].values
        y = self.df.loc[self.VOLAT_PERIOD:self.df.shape[0] - self.PERIOD - 1, "return"].values
        X = np.vstack((x, y)).T
        self.model.fit(X)
        labels = self.model.labels_
        clusterCenters = self.model.cluster_centers_
        colors = cm.rainbow(np.linspace(0, 1, self.model.n_clusters))

        self.logger.info("Cluster centeres")
        self.logger.info(clusterCenters)

        for i in range(self.model.n_clusters):
            xlab = x[labels == i]
            ylab = y[labels == i]
            plt.scatter(xlab, ylab, c=colors[i], label=str(i))
            plt.scatter([clusterCenters[i, 0]], [clusterCenters[i, 1]], c='black')

        plt.grid()
        plt.xlabel("Volatility")
        plt.ylabel("5-Day Return")
        plt.title("Scatterplot of Classified Points")
        plt.legend()
        plt.savefig(os.path.join(self.dirname, f"scatterplot_classified_{self.__class__.__name__}.jpeg"),
                    dpi=500)
        plt.show()



if __name__ == "__main__":
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    kmeans = KMeansCluster(dirname, "SPY")
    kmeans.fit()
