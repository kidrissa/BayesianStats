import os
import sys
from numpy import (
    array, ndarray, zeros, ones, vstack,
    trace, exp, pi, sqrt, log
)
from pandas import DataFrame
import statsmodels.api as sm
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from typing import Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import weighted_least_square_error


class ModelBasedClustering:
    def __init__(self, data: DataFrame, K: int) -> None:
        self.X = data.iloc[:, 2:].values
        self.y = data.iloc[:, [1]].values.reshape(-1)
        self.nClusters = K
        self.Theta = zeros(K)
        self.Beta = zeros(data.iloc[:, 2:].shape[1])
        self.Sigma = 0
        self.Tau = zeros(K)
        self.Z = ones((data.shape[0], K))
        self.Z_hat = ones((data.shape[0], K))

    def initZ(self) -> ndarray:
        """initialize the Z vector using the ecommeded method:
        the k-means algorithm applied to the regression residuals.

        Returns:
            ndarray: initial matrix Z
        """
        OLS_residuals = sm.OLS(self.y, sm.add_constant(self.X)).fit().resid
        OLS_residuals = OLS_residuals.reshape(len(OLS_residuals), 1)
        kmeans = KMeans(
            n_clusters=self.nClusters, random_state=42
        ).fit(OLS_residuals)
        clustersVect = kmeans.labels_
        return vstack(
            [(clustersVect == i).astype(int) for i in range(self.nClusters)]
        ).T

    def eval_Tau(self) -> ndarray:
        """compute the vector of tau at a given step

        Returns:
            ndarray: Tau array
        """
        return array([
            self.Z_hat[:, j].mean() for j in range(self.Z_hat.shape[1])
        ])

    def eval_Sigma(self) -> float:
        """Compute the estimator of standard errors Sigma at a given step

        Returns:
            float: std error Sigma
        """
        ZHat, Theta_, Beta_ = self.Z_hat, self.Theta, self.Beta
        X_, y_ = self.X, self.y
        n_, K_ = self.y.shape[0], self.nClusters

        error_matrix = array(
            [
                [(y_[i] - Theta_[j] - X_[i].dot(Beta_))**2 for j in range(K_)]
                for i in range(n_)
            ]
        )
        return trace(ZHat.dot(error_matrix.T)) / n_

    def eval_Zhat(self) -> ndarray:
        """Compute Zhat at a given iteration

        Returns:
            ndarray: Z estimation
        """
        tau_, sigma_ = self.eval_Tau(), self.eval_Sigma()
        y_, X_ = self.y, self.X
        theta_, beta_ = self.Theta, self.Beta

        def phi_(i: int, j: int) -> float:
            """Evaluate phi the content in the exponential

            Args:
                i (int): row index, representing an obs
                j (int): column index, representing a cluster

            Returns:
                float: value of phi
            """
            return tau_[j] * exp(
                -(1 / (2 * sigma_)) * (y_[i] - theta_[j] - X_[i].dot(beta_))**2
            )

        def phi_normlize(i: int) -> float:
            """Evaluate the constante bof normalization of phis

            Args:
                i (int): row index, representing an obs

            Returns:
                float: phis constante of normalization
            """
            return array([phi_(i, j) for j in range(self.nClusters)]).sum()

        return array(
            [
                [phi_(i, j) / phi_normlize(i) for j in range(self.nClusters)]
                for i in range(y_.shape[0])
            ]
        )

    def Zhat_to_Z(self) -> ndarray:
        """Get Z from Zhat, the estimation of Z

        Returns:
            ndarray: matrix Z
        """
        ZHat = self.Z_hat
        return vstack([
            (ZHat[i] == ZHat.max(axis=1)[i]).astype(
                int) for i in range(ZHat.shape[0])
        ])

    def minimizaton(self) -> ndarray:
        """solve the minimizaion problem using the scipy solver at
        a given step

        Returns:
            ndarray: optimal values of the problem
        """
        ZHat, X_, y_ = self.Z_hat, self.X, self.y
        p_, K_ = self.X.shape[1], self.nClusters
        x0 = ones(p_ + K_)
        res = minimize(
            weighted_least_square_error,
            x0,
            method="nelder-mead",
            args=(ZHat, X_, y_, K_),
            options={"xatol": 1e-8, "disp": False},
        )
        return res.x

    def EM_algo(self, Niter: int) -> Tuple[ndarray, float]:
        """Run the EM algorithm

        Args:
            Niter (int): number of iteration

        Returns:
            Tuple[ndarray, float]: _description_
        """
        """
        solve the minimisation program
        compute the model likelihood for BIC evaluation
        compute the BIC value
        """
        ZPrev, ZHatPrev = self.initZ(), self.initZ()
        for _ in range(Niter):
            # initialization
            self.Z_hat, self.Z = ZPrev, ZHatPrev

            # Minimizaton problem solving
            stacked_Theta_Beta = self.minimizaton()
            self.Theta = stacked_Theta_Beta[: self.nClusters]
            self.Beta = stacked_Theta_Beta[self.nClusters:]

            # Parameters updating
            self.Sigma = self.eval_Sigma()
            self.Tau = self.eval_Tau()
            self.Z_hat = self.eval_Zhat()
            self.Z = self.Zhat_to_Z()
            if (self.Z == ZPrev).all():
                break
            else:
                ZPrev, ZHatPrev = self.Z, self.Z_hat
        y_, X_ = self.y, self.X
        Theta_, Beta_ = self.Theta, self.Beta
        n_, K_ = self.y.shape[0], self.nClusters

        def normal_pdf(x: ndarray) -> float:
            """Standard gaussian law pdf

            Args:
                x (ndarray): input

            Returns:
                float: densiy value
            """
            return (1 / sqrt(2 * pi)) * exp(-(0.5) * x**2)

        likelihood = array(
            [
                [
                    (1 / sqrt(self.eval_Sigma()))
                    * normal_pdf(
                        (y_[i] - Theta_[j] - X_[i].dot(Beta_))/sqrt(
                            self.eval_Sigma()
                        )
                    )
                    for j in range(K_)
                ]
                for i in range(n_)
            ]
        )

        return self.Z, - 2 * log(likelihood.dot(self.eval_Tau()).prod()) + (
            2 * self.nClusters + self.X.shape[1]
        ) * log(self.X.shape[0])
