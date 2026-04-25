import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X)
    y = np.array(y)

    # Check X shape

    assert X.shape[0] >= X.shape[1], "X must have at least as many rows as columns (n >= d)"

    # compute tanspose of X
    X_t = X.T

    # compute products of transpose with X and y
    X_t_X = X_t @ X

    X_t_y = X_t @ y

    # compute the inverse of X_t_x
    X_inverse = np.linalg.pinv(X_t_X)

    w = X_inverse @ X_t_y

    assert len(w) == X.shape[1], "Lenght w does not equal number of columns in X"
    return w