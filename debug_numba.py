import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numba
import numpy as np
from sklearn.datasets import make_regression


@numba.jit(nopython=True, cache=False, parallel=True)
def stable_fit(X, y, threshold=3):
    """Fitting stable regressions using an adapted CCDC method

    Models are first fit using OLS regression. Those models are then checked for
    stability. If a model is not stable, the two oldest
    acquisitions are removed, a model is fit using this shorter
    time-series and again checked for stability. This process continues as long
    as all of the following 3 conditions are met:

    1. The timeseries is still unstable
    2. There are enough cloud-free acquisitions left (threshold is 1.5x the
        number of parameters in the design matrix)
    3. The time series includes data of more than half a year

    Stability depends on all these three conditions being true:
    1.             slope / RMSE < threshold
    2. first observation / RMSE < threshold
    3.  last observation / RMSE < threshold

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        threshold (float): Sensitivity of stability checking.

    Returns:
        beta (numpy.ndarray): The array of regression estimators
        residuals (numpy.ndarray): The array of residuals
        is_stable (numpy.ndarray): 1D Boolean array indicating stability
    """
    min_obs = int(X.shape[1] * 1.5)
    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    residuals = np.full_like(y, np.nan)
    stable = np.empty((y.shape[1]))
    for idx in numba.prange(y.shape[1]):
        y_sub = y[:, idx]
        isna = np.isnan(y_sub)
        X_sub = X[~isna]
        y_sub = y_sub[~isna]
        is_stable = False

        # Run until minimum observations
        # or until stability is reached
        for jdx in range(len(y_sub), min_obs-1, -2):
            # Timeseries gets reduced by two elements
            # each iteration
            y_ = y_sub[-jdx:]
            X_ = X_sub[-jdx:]
            beta_sub = np.linalg.solve(np.dot(X_.T, X_), np.dot(X_.T, y_))
            resid_sub = np.dot(X_, beta_sub) - y_
            # Check for stability
            rmse = np.sqrt(np.mean(resid_sub ** 2))
            slope = np.fabs(beta_sub[1]) / rmse < threshold
            first = np.fabs(resid_sub[0]) / rmse < threshold
            last = np.fabs(resid_sub[-1]) / rmse < threshold
            # Break if stability is reached
            is_stable = slope & first & last
            if is_stable:
                break

        beta[:,idx] = beta_sub
        residuals[-jdx:,idx] = resid_sub
        stable[idx] = is_stable
    return beta, residuals, stable.astype(np.bool_)

if __name__ == '__main__':
# Run on test data
    np.random.seed(8)
    X, y = make_regression(n_samples=200, n_features=4, n_targets=50)
    stable_fit(X,y)
