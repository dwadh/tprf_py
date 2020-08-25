import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score as rr2
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA


def tprf(X, y, Z, oos_present, oos=[]):
    """
    Computes the returns (y) based on the set of predictors
    (X) and proxies (Z) using the three pass regression filter method

    :param X: Array of predictors (Shape: T x N, where
    T = number of timestamps in the training set,
    N = number of predictors)
    :param y: Array of returns (Shape: T x 1, where
    T = number of timestamps in the training set)
    :param Z: Array of proxies (Shape: T x L, where
    T = number of timestamps in the training set,
    L = number of proxies)
    :param oos_present: True if out of sample data is present (oos),
    False otherwise
    :param oos: Out of sample data (predictors)(Shape: 1 x N)
    :return yhat: Forecasted returns for in sample data (Shape: T x 1)
    :return yhatt: Forecasted return for out of sample array of predictors (float)
    """

    # Pass 1 (Dependent - Value of predictor i across given time intervals,
    # Independent - Set of proxies)

    phi = np.ndarray(shape=(X.shape[1], Z.shape[1]))
    eta = []
    for i in range(0, X.shape[1]):
        first_pass_model = LR()
        first_pass_model = first_pass_model.fit(X=Z, y=X[:, i])
        phi[i, :] = first_pass_model.coef_
        eta.append(first_pass_model.intercept_)

    # Pass 2 (Dependant - Cross section of predictor values at time t,
    # Independent - phi (from Pass 1)

    eta = np.array(eta).reshape(X.shape[1], 1)
    sigma = np.ndarray(shape=(X.shape[0], Z.shape[1]))
    eta1 = []
    for t in range(0, X.shape[0]):
        second_pass_model = LR()
        second_pass_model.fit(X=phi, y=X[t, :].T)
        sigma[t, :] = second_pass_model.coef_.flatten()
        eta1.append(second_pass_model.intercept_)

    eta1 = np.array(eta1)

    # Pass 3 (Dependant - Array of returns, Independent - sigma (from Pass 2)

    third_pass_model = LR()
    third_pass_model.fit(X=sigma, y=y)
    coeff, intercept = (third_pass_model.coef_, third_pass_model.intercept_)
    yhat = np.dot(sigma, coeff.T) + intercept

    # If out of sample set of predictors is present, compute the forecasted
    # return by running the second pass with out of sample predictors as the
    # dependant variable, and multiplying the resultant sigma with beta (coeff) from
    # the previous third pass and adding the intercept

    yhatt = np.nan
    if oos_present:
        second_pass_model = LR()
        second_pass_model.fit(X=phi, y=oos)
        sigma = second_pass_model.coef_.flatten()
        yhatt = np.dot(sigma, coeff.T) + intercept
    return yhat, yhatt


def autoproxy(X, y, n_proxy):
    """
    Use the autoproxy algorithm for calculating the proxies,
    given an array of predictors and corresponding target values

    :param X: Array of predictors
    :param y: Array of returns (target values)
    :param n_proxy: number of proxies to be calculated
    :return r0: Array of proxies
    """
    r0 = np.array(y)
    yhatt = 1
    for i in range(0, n_proxy - 1):
        (yhat, yhatt) = tprf(X, y, r0, False)
        r0 = np.hstack([y - yhat, r0])
    return r0


def recursive_train(X, y, Z, train_window):
    """
    Recursively train on the training data and predict on the
    out of sample data using the tprf model
    :param X: Array of predictors
    :param y: Array of returns (target)
    :param Z: If int: number of proxies to be calculated
    If array: Array of proxies (Shape: TxL)
    :param train_window: Initial training size to be used
    :return: R2 score of the fit
    """
    lst = []
    if isinstance(Z, int):
        do_autoproxy = True
        n_proxies = Z
    else:
        do_autoproxy = False
    for t in range(train_window, X.shape[0]):
        #print("Train on -> 0:", t - 1)
        #print("Test on -> ", t)
        if do_autoproxy:
            Z = autoproxy(X[:t], y[:t].reshape(-1, 1), n_proxies)
        else:
            Z = Z[:t]
        X_train = X[:t]
        X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
        X_test = X[t]
        X_test = (X_test.T/np.std(X_test, axis = 0).reshape(-1, 1).flatten()).T
        y_train = y[:t]
        yhat, yhatt = tprf(X_train, y_train, Z, True, X_test)
        lst.append(yhatt)
    yhatt = np.array(lst)
    y_true = y[train_window:]
    return rr2(y_true, yhatt)


def pclas(X_train, y_train, X_test, n_components):
    """
    Function to implement the PCLAS procedure
    :param X_train: Array of predictors to train the model on
    :param y_train: Array of returns corresponding to X_train
    :param X_test: Out of sample array of predictors to predict the return on
    :param n_components: Number of principal components to be used
    :return:
    """
    model = Lasso(alpha = 0.1).fit(X_train, y_train)
    model = SelectFromModel(model, prefit = True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    yhat = pcr(X_train, y_train, X_test, n_components)
    return (yhat)


def pcr(X_train, y_train, X_test, n_components):
    """
    Function to implement the PCR procedure
    :param X_train: Array of predictors to train the model on
    :param y_train: Array of returns corresponding to X_train
    :param X_test: Out of sample array of predictors to predict the return on
    :param n_components: Number of principal components to be used
    :return yhat: Predicted return
    """
    pca = PCA(n_components = n_components)
    pca = pca.fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)
    model = LR()
    model = model.fit(X = Z_train, y = y_train)
    coef = model.coef_
    intercept = model.intercept_
    yhat = np.dot(coef, Z_test.T) + intercept
    return yhat


def recursive_train_alternate(X, y, train_window, n_components, procedure):
    """
    Recursively train on the training data and predict on the
    out of sample data using the pcr, pclas procedure
    :param X: Array of predictors
    :param y: Array of returns (target)
    :param train_window: Initial training size to be used
    :param n_components: Number of principal components to be used
    :param procedure: pcr or pclas
    :return: R2 score of the fit
    """
    r_lst = []
    lst = []
    for t in range (train_window, X.shape[0]):
        X_train = X[:t]
        X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
        X_test = X[t]
        X_test = (X_test.T/np.std(X_test, axis = 0).reshape(-1, 1).flatten()).T
        y_train = y[:t]
        if (procedure == "pclas"):
            yhatt = pclas(X_train, y_train, X_test.reshape(1, -1), n_components)
        elif (procedure == "pcr"):
            yhatt = pcr(X_train, y_train, X_test.reshape(1, -1), n_components)
        lst.append(yhatt)
    yhatt = np.array(lst)
    y_true = y[train_window:]
    return (rr2(y_true, yhatt))
