from numpy import array, zeros, vstack, exp
from pandas import DataFrame, read_csv
from os import chdir

import statsmodels.api as sm
from sklearn.cluster import KMeans

chdir("C:/Users/idris/Desktop/ENSAE/S1_3A/Statistique_Bayesienne/BayesianStats/")

class mb_clustering:
    def __init__(self, data:DataFrame, K:int) -> None:
        self.X = data.iloc[:, 2:]
        self.y = data.iloc[:, [1]]
        self.nClusters = K
        self.theta = zeros(K)
        self.beta = zeros(data.shape[1])
        self.Z = zeros((data.shape[1], K))

    
    def initZ(self):
        """
        initialize the Z vector using the ecommeded method:
        the k-means algorithm applied to the regression residuals.
        """
        OLS_residuals = sm.OLS(self.y, sm.add_constant(self.X)).fit().resid.values
        OLS_residuals = OLS_residuals.reshape(len(OLS_residuals), 1)

        kmeans = KMeans(n_clusters=self.nClusters, random_state=0).fit(OLS_residuals)
        clustersVect = kmeans.labels_
        Z = vstack([(clustersVect == i).astype(int) for i in range(self.nClusters)])

        return Z.reshape(Z.shape[1], Z.shape[0])


    def tau(self):
        Z_ = self.Z
        return array([Z_[:, j].mean() for j in range(Z_.shape[1])])


    def sigma(self):
        Zhat = self.Zhat()


    def Zhat(self):
        """
        compute Zhat at a given iteration
        """
        tau_, sigma_ = self.tau(), self.sigma()
        y_, X_ = self.y, self.X
        theta_, beta_ = self.theta, self.beta
        phi_ = lambda i, j: tau_[j] * exp(-(1/(2*sigma_**2))*(y_[i] - theta_[j] - X_[i]*beta_)**2)
        phi_normlize = lambda i: array([phi_(i, j) for j in range(self.nClusters)]).sum()
        return array([[phi_(i, j)/phi_normlize(i) for j in range(self.nCluster)] for i in range(y_.shape[0])])


    def log_likelihood(self):
        """
        evaluate the likelihood of the model
        """


    #def EM_algo(self):


data_happiness = read_csv("./data/world_data_processed.csv", sep=";")
print(mb_clustering(data_happiness, 5).initZ())