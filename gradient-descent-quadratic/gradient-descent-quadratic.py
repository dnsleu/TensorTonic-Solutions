import numpy as np

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    f(x)=ax**2 +bx + c
    Return final x after 'steps' iterations.
    """
    # Function
    p = np.polynomial.Polynomial([c,b,a])

    # First derivative
    d = p.deriv(1)
    target = d.roots().tolist()[0]

    # Lists for training
    step_count = []
    loss_val = []

    for step in range(steps):
        # 1. Forward pass
        slope = np.round(d(x0),3)
        loss = d(target) - slope
        loss = np.round(loss,3)

        # 2. Update
        x0 = x0 - lr * d(x0)
        
        # 3. Print results
        step_count.append(step)
        loss_val.append(loss)
        print(f"Step: {step+1} | x: {x0:.6f} | Loss: {loss:.6f} | Slope = {slope:.6f}")

        # 4. Exit loop when result found
        if loss == 0.000:
            break
        
    return x0
    pass