def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X = np.array(X)
    W = np.array(W)
    b = np.array(b)

    assert X.shape[0] > 0 and X.shape[1] > 0, "X must have at least 1 row and 1 column"

    y = np.zeros((X.shape[0], len(b)))
    
    if W.shape == (X.shape[1], len(b)):
        for i in range(len(X)):
            for j in range(len(b)):
                y[i][j] = sum(X[i][k] * W[k][j] for k in range(X.shape[1])) + b[j]
                #print(f"y: {y[i][j]}, Dot: {sum(X[i][k] * W[k][j] for k in range(X.shape[1]))}, b: {b[j]}")

    elif W.shape == (len(b), X.shape[1]):
        for i in range(len(X)):
            for j in range(len(b)):
                    y[i][j] = sum(X[i][k] * W[j][k] for k in range(X.shape[1])) + b[j]
                    #print(f"y: {y[i][j]}, Dot: {sum(X[i][k] * W[j][k] for k in range(X.shape[1]))}, b: {b[j]}")
    else:
        raise ValueError("Shapes are incompatible")

    return y.tolist()