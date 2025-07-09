import numpy as np
from scipy.integrate import odeint

class LogisticMap:
    def __init__(self, r=3.9, x=0.5):
        self.r = float(r)
        self.x = x
        self.history = []
        self.type = "LOGISITC"
        self.lyapunov_exponent = 0.0

        self.initial_state = [0.5,]
        self.params = [self.r,]

    def step(self):
        self.x = self.r * self.x * (1 - self.x)
        self.history.append(self.x)
        if len(self.history) > 500:
            self.history.pop(0)
        #return self.x, 1 - self.x, self.x * (1 - self.x)
        return self.x, 0.0, 0.0

    def get_history(self):
        return [np.array(self.history)]
    
    def get_state(self):
        return (self.x,)
    
    def get_params(self):
        return {'r':self.r}
    
    def get_expected_le_range(self):
        return (0.0, 0.7)
    
    def map(self, state):
        # Ensure state is unpacked and cast to float
        x = float(state[0]) if isinstance(state, (list, tuple, np.ndarray)) else float(state)
        r = float(self.r)
        x_new = r * x * (1.0 - x)
        return np.array([x_new])
    
    def get_map_function(self):
        def f(state, r):
            x = float(state[0]) if isinstance(state, (list, tuple, np.ndarray)) else float(state)
            r = float(self.r)
            x_new = r * x * (1.0 - x)
            return np.array([x_new])
        return f

class HenonMap:
    def __init__(self, a=1.4, b=0.3):
        self.a = float(a)
        self.b = float(b)
        self.x = 0.1
        self.y = 0.1
        self.history_x = []
        self.history_y = []
        self.type = "HENON"
        self.lyapunov_exponent = 0.0
        self.parameters = [self.a, self.b]

        self.initial_state = [self.x, self.y]

    def step(self):
        new_x = 1 - self.a * self.x ** 2 + self.y
        new_y = self.b * self.x
        self.x, self.y = new_x, new_y
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        if len(self.history_x) > 500:
            self.history_x.pop(0)
            self.history_y.pop(0)
        #return self.x, self.y, self.x * self.y
        return self.x, 0.0, 0.0 

    def get_history(self):
        return [np.array(self.history_x), np.array(self.history_y)]
    
    def get_state(self):
        return (self.x, self.y)
    
    def get_params(self):
        return {'a':self.a, 'b':self.b}
    
    def get_expected_le_range(self):
        return (0.3, 0.6)
    
    def map(self, state, a, b):
        x, y = map(float, state)
        x_new = 1.0 - a * x * x + y
        y_new = b * x
        return (x_new, y_new)
    
    def get_map_function(self):
        def f(state, a, b):
            x, y = state
            x_new = 1 - a * x * x + y
            y_new = b * x
            return np.array([x_new, y_new])
        return f

class LorenzAttractor:
    def __init__(self, sigma=10, rho=28.0, beta=(8/3), dt=0.015):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.history = []
        self.type = "LORENZ"
        self.lyapunov_exponent = 0.0
        self.parameters = (self.sigma, self.rho, self.beta)

        self.initial_state = np.array([1.0, 1.0, 1.0])

    def step(self):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        self.history.append((self.x, self.y, self.z))
        if len(self.history) > 500:
            self.history.pop(0)
        return self.x*2, self.y*2, self.z*2

    def get_history(self):
        return zip(*self.history[-2000:])
    
    def get_state(self):
        return (self.x, self.y, self.z)
    
    def get_params(self):
        return{'sigma':self.sigma, 'rho':self.rho, 'beta': self.beta}
    
    def get_expected_le_range(self):
        return (0.90, 1.50)
    
    def solve_system(self, state, sigma, rho, beta):
        x, y, z = state

        max_val = 1e6
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x_new = np.clip(x + dx * self.dt, -max_val, max_val)
        y_new = np.clip(y + dy * self.dt, -max_val, max_val)
        z_new = np.clip(z + dz * self.dt, -max_val, max_val)

        return [x_new, y_new, z_new]
    
    def get_map_function(self):
        def f(state, sigma, rho, beta):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])
        return f
    
class DuffingOscillator:
    def __init__(self):
        self.alpha = 1.0    # Linear stiffness
        self.beta = 0.5     # Nonlinear stiffness
        self.delta = 0.1    # Damping
        self.gamma = 0.3    # Forcing amplitude
        self.omega = 1.2    # Forcing frequency

        self.type = "DUFFING"
        self.lyapunov_exponent = 0.0
        self.parameters = self.get_params()

        self.x = 0.1 + np.random.normal(0, 0.1) 
        self.v = 0.0
        self.t = 0.0
        self.initial_state = [0.1, 0.0]
        self.t_span = np.linspace(0, 10, 1000)
        self.history = []

        self.num_steps = 20

    def step(self):
        sol = odeint(
            self.solve_system,
            self.initial_state,
            self.t_span,
            args=(self.delta, self.alpha, self.beta, self.gamma, self.omega)
        )

        x_vals = sol[:, 0]
        v_vals = sol[:, 1]

        # Update internal state to last known values
        self.initial_state = [x_vals[-1], v_vals[-1]]

        # Return final state (audio-rate point)
        return x_vals[-1], v_vals[-1], self.t_span[-1]

    def get_history(self):
        if not self.history:
            return [np.array([]), np.array([]), np.array([])]
        return [np.array(x) for x in zip(*self.history[-2000:])]
    
    def modulate_params(self, time, velocity=1.0):
        self.alpha = -1.0 + 0.5 * np.sin(0.1 * time)
        self.beta = 1.0 + 0.5 * np.cos(0.05 * time)
        self.gamma = 0.3 + 0.2 * velocity
        self.omega = 1.2 + 0.1 * np.sin(0.07 * time)

    def get_state(self):
        return (self.x, self.v)
    
    def get_params(self):
        return{
            'alpha': self.alpha,
            'beta':self.beta,
            'delta':self.delta,
            'gamma':self.gamma,
            'omega':self.omega
        }
    
    def get_expected_le_range(self):
        return [0.1, 1.0]
    
    def solve_system(self, state, t, delta, alpha, beta, gamma, omega):
        x, v = state
        dxdt = v
        dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        return [dxdt, dvdt]

