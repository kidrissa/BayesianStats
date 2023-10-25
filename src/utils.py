from numpy import (
    array, trace, ndarray
)


def weighted_least_square_error(
        x: ndarray, ZHat: ndarray, X_: ndarray,
        y_: ndarray, K_: ndarray
) -> float:
    """compute the weighted least squares function:

    Args:
        x (ndarray): stack of vectors of parameters Theta_,
        Beta_ of length K+p
        ZHat (ndarray): _description_
        X_ (ndarray): _description_
        y_ (ndarray): _description_
        K_ (ndarray): _description_

    Returns:
        float: square error
    """
    n_ = X_.shape[0]

    Theta_, Beta_ = x[:K_], x[K_:]
    error_matrix = array(
        [
            [(y_[i] - Theta_[j] - X_[i].dot(Beta_))**2 for j in range(K_)]
            for i in range(n_)
        ]
    )
    return trace(ZHat.dot(error_matrix.T))
