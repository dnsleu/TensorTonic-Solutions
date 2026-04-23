import numpy as np
import matplotlib.pyplot as plt

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    b = 0
    w = np.zeros(X.shape[1])

    losses = []
    
    for step in range(steps+1):
        # define z function
        z = X.dot(w) + b
    
        # Predict using X
        p = _sigmoid(z)
    
        # Compute loss
        loss = -np.round((y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),3)
        losses.append(loss)
    
        # Calculate derivatives
        dW = X.T.dot(p-y) / len(X)
        dB = (p-y).mean()
    
        # Update the parameters
        w -= lr * dW
        b -= lr * dB

        if step % 50 == 0:
            print(f"\n\n Step: {step} | Weight: {np.array2string(w, precision=3)} | Bias: {b:.3f} | x: {X.T} | p: {np.array2string(p, precision=3)} | y: {y} | Loss: {loss:.3f} | dW: {np.array2string(dW, precision = 3)} | dB: {dB:.3f}")
    # Exit loop when loss 0
        if loss == 0.000:
            break
            
    fig,ax = plt.subplots(figsize = (10,8))

    ax.plot(np.arange(steps+1), losses, lw = 2, ls = "-", color = "orange", label = f"Training loss (final = {loss:.4f})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Logisitc regression - training")
    plt.legend(loc = "upper right")
    plt.show()
    
    if X.shape[1] == 1:
        x_grid = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 200)
        grid = np.c_[x_grid]
        probs = _sigmoid(grid.dot(w) + b)
        
        plt.figure()
        plt.plot(x_grid, probs, label="Predicted probability")
        plt.scatter(X[:, 0], y, c=y, cmap="Greens", edgecolor="k")
        plt.axhline(0.5, linestyle="--")
        plt.title("Logistic regression - decision boundary")
        plt.xlabel("X")
        plt.ylabel("Probability / class")
        plt.show()
    
    return w, b
    
    pass