import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    X = np.array(A)
    
    assert 1 <= X.shape[0] and X.shape[1] <= 100, "X must be 2 dimensional, with sizes ranging from 1 to 100"

    # 1) X = np.permute_dims(X)
    # 2) X = np.array(list(map(list, zip(*X))))
    
    result = np.zeros_like(X)
    result = result.reshape(X.shape[1], X.shape[0])

    # loop through rows
    for i in range(len(X)):
        # loop through columns
        for j in range(len(X[0])):
            result[j][i] = X[i][j]
            
    result = np.array(result)

    return result
    
    pass
