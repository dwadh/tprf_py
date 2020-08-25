import numpy as np


def data_generator(data_dict):
    """
    Function to generate data samples, governed by the parameters passed through
    the data_dict, based on the setup mentioned in the paper.
    :param data_dict: dictionary consisting of values:
    T: Number of time-points
    N: Number of predictors
    pf: Serial correlation in relevant factors
    pg: Serial correlation in irrelevant factors
    d: Cross correlation parameter for Cross-sectional correlation among idiosyncrasies
    a: Serial correlation between idiosyncratic errors
    sigma_y: Multiplicative factor for uncertainty in returns
    relevant_factors: Number of relevant factors (f) to use
    irr_factors: Number of irrelevant factors (g) to use
    The variables used above and in the code follow the same notation as the ones used in the paper
    :return X: Array of predictor values over a tome period (Shape: TxN)
    :return y: Array of returns (Shape: Tx1)
    """

    # Load the values from the data dictionary
    T = data_dict["T"]
    N = data_dict["N"]
    pf = data_dict["pf"]
    pg = data_dict["pg"]
    d = data_dict["d"]
    a = data_dict["a"]
    strength = data_dict["strength"]
    v = np.random.normal(size=(T, N))

    # Calculate the size of factor loadings and draw them as standard normals
    n_factor_loadings = data_dict['relevant_factors'] + data_dict['irr_factors']
    factor_loadings = np.random.normal(size=(n_factor_loadings, N))
    if data_dict['non_pervasive']:
        set_zero_index = np.random.choice(N, replace=False, size = int(N/2))
        factor_loadings[0, [set_zero_index]] = 0

    # Generate the relevant (f) and irrelevant (g) factors
    f = np.ndarray(shape=(T, data_dict['relevant_factors']))
    g = np.ndarray(shape=(T, data_dict['irr_factors']))
    f[0] = np.random.normal()
    # g_var: required higher variance of irrelevant factors
    g_var = [1.25, 1.75, 2.25, 2.75]
    for t in range(1, T):
        f[t] = f[t - 1] * pf + np.random.normal(loc=0, scale=1)

    g[0, :] = np.random.normal()
    g_err = np.zeros(shape=(T, data_dict['irr_factors']))
    for i in range (0, 4):
        g_err[:, i] = np.random.normal(size=(T,), loc=0, scale=1)
    for j in range (0, data_dict['irr_factors']):
        for t in range (1, T):
            g[t][j] = g[t-1][j] * pg + g_err[t][j]

        # Adjust the irrelevant factors for higher variance than relevant factors
        g[:, j] = (g[:, j]/np.std(g[:, j]))*f.std()*np.sqrt(g_var[j])

    # Generate the returns y
    y = np.ndarray(shape=(T + 1,))
    sigma_y = np.std(f)
    for t in range(1, T + 1):
        y[t] = f[t - 1] + sigma_y * np.random.normal(loc=0, scale=1)

    # Generate the idiosyncratic errors
    eta_tilda = np.ndarray(shape=(T, N))
    eta = np.ndarray(shape=(T, N))
    for t in range(0, T):
        for i in range(0, N):
            if i == (N - 1):
                eta_tilda[t][i] = ((1 + d ** 2) * v[t][i]) + (d * v[t][i - 1]) + (d * v[t][0])
                continue
            eta_tilda[t][i] = ((1 + d ** 2) * v[t][i]) + (d * v[t][i - 1]) + (d * v[t][i + 1])

    eta[0][:] = np.random.normal(size=(N,))
    for t in range(1, T):
        for i in range(0, N):
            eta[t][i] = a * eta[t - 1][i] + eta_tilda[t][i]

    # Multiply the factor loadings with the factors and add the idiosyncratic
    # errors to generate the predictors
    factors = np.hstack([f, g])
    # constant is the multiplicative constant to adjust the dependence of
    # predictor variance on factors for normal, moderate, weak factors
    X1 = np.dot(factors, factor_loadings)
    eta = eta/np.std(eta)
    constant = np.std(X1) * np.sqrt(strength)
    eta = constant*eta
    X = X1 + eta
    y = y[1:]

    return (X, y)
