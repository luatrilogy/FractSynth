import numpy as np
from scipy.integrate import odeint

def _step_continuous(system_func, state, params, dt):
    t = [0, dt]
    sol = odeint(system_func, state, t, args=params)
    return sol[-1]  # final state after dt

def _step_discrete(system_func, state, params):
    return np.array(system_func(state, *params))

def _lyapunov_continuous(attractor, initial_state, params, epsilon=1e-8, steps=10000, dt=0.01):
    x1 = np.array(initial_state, dtype=np.float64)
    x2 = x1 + np.random.normal(scale=epsilon, size=x1.shape)

    sum_lyap = 0.0
    for _ in range(steps):
        f1 = np.array(attractor(x1, *params), dtype=np.float64)
        f2 = np.array(attractor(x2, *params), dtype=np.float64)

        x1 = x1 + dt * f1
        x2 = x2 + dt * f2

        delta_vec = x2 - x1
        delta = np.linalg.norm(delta_vec)
        if delta < 1e-12 or np.isnan(delta):
            continue

        sum_lyap += np.log(delta / epsilon)

        # Renormalize perturbed point to be distance epsilon from x1
        x2 = x1 + epsilon * (delta_vec / delta)

    average_le = sum_lyap / (steps * dt)
    return average_le, [average_le] * steps

def _lyapunov_discrete(system_func, initial_state, params, epsilon=1e-8, steps=int(1000)):
    x1 = np.array(initial_state, dtype=np.float64)
    x2 = x1 + epsilon * np.random.normal(size=len(x1))

    le_sum = 0.0
    le_values = []

    for _ in range(steps):
        x1 = _step_discrete(system_func, x1, params)
        x2 = _step_discrete(system_func, x2, params)

        delta = np.linalg.norm(x2 - x1)
        if delta < 1e-12:
            delta = 1e-12

        le_step = np.log(delta / epsilon)
        le_sum += le_step
        le_values.append(le_sum / (_ + 1))

        # Renormalize perturbed state
        x2 = x1 + epsilon * (x2 - x1) / delta

    return le_sum / steps, le_values

# --- High-Level Wrappers ---

def calculate_lyapunov(attractor, func, steps=1000, dt=0.01):
    """
    General-purpose LE calculator that adapts to both continuous and discrete systems.
    """
    is_discrete = hasattr(attractor, 'map')  # e.g., LogisticMap, HenonMap
    params = getattr(attractor, 'parameters', ())
    initial_state = np.array(attractor.initial_state, dtype=np.float64)

    if is_discrete:
        func = attractor.map
        le, le_values = _lyapunov_discrete(func, initial_state, params, steps=steps)
    else:
        func = attractor.solve_system
        le, le_values = _lyapunov_continuous(func, initial_state, params, steps=steps, dt=dt)

    return le, le_values