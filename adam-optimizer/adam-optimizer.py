import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Step count
    t_count = []
    
    for t in range(1,t+1,1):
        # 1. Update first moment (momentum):
        m_new = np.array(beta1) * m + np.array((1-beta1)) * grad

        # 2. Update second moment (adaptive rate / velocity)
        v_new = np.array(beta2) * v + np.array((1-beta2)) * np.power(grad,2)

        # 3. Bias correction
        m_hat = m_new / (1 - beta1 ** t)

        v_hat = v_new / (1 - beta2 ** t)

        # 4. Update parameters
        param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

        # Print training
        t_count.append(t)
        print(f"Step: {t}, Parameters: {param_new}, Moment: {m_new}, Velocity: {v_new}, M_hat: {m_hat}, V_hat: {v_hat}")

    return param_new, m_new, v_new
    pass