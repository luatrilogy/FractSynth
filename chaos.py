import numpy as np
from scipy.integrate import odeint

_HAS_CUPY = False
_xp = np

try:
    import cupy as _cp
    try:
        # Only enable if at least one CUDA device is usable
        if _cp.cuda.runtime.getDeviceCount() > 0:
            _xp = _cp
            _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
except Exception:
    _HAS_CUPY = False

def _to_xp(arr):
    if _HAS_CUPY:
        return _cp.asarray(arr)
    return np.asarray(arr)

def _to_numpy(arr):
    if _HAS_CUPY:
        return _cp.asnumpy(arr)
    return np.asarray(arr)

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
    
    def step_batch(self, frames):
        x = float(self.x)
        r = float(self.r)
        xs = np.empty(frames, dtype=np.float32)
        ys = np.empty(frames, dtype=np.float32)
        zs = np.empty(frames, dtype=np.float32)

        prev = x
        for i in range(frames):
            x = r * x * (1.0 - x)
            xs[i] = x
            ys[i] = x - prev          # simple derivative proxy → nonzero amp
            zs[i] = (i / max(1, frames-1)) * 2.0 - 1.0  # sweep pan [-1,1]
            prev = x

            self.history.append(x)
            while len(self.history) > 500:
                self.history.pop(0)

        self.x = float(x)
        return xs, ys, zs

    def get_history(self):
        return (np.asarray(self.history, np.float32),)
    
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
    
    def step_batch(self, frames):
        xp = np  # we’ll keep CPU here; GPU adds no benefit for this tiny loop
        a = self.a; b = self.b
        x = float(self.x); y = float(self.y)

        xs = xp.empty(frames, dtype=xp.float32)
        ys = xp.empty(frames, dtype=xp.float32)
        zs = xp.zeros(frames, dtype=xp.float32)  # pan can be 0 if you like

        for i in range(frames):
            nx = 1.0 - a * x * x + y
            ny = b * x
            x, y = nx, ny
            xs[i] = x; ys[i] = y

            self.history_x.append(float(x))
            self.history_y.append(float(y))
            while len(self.history_x) > 500:
                self.history_x.pop(0); self.history_y.pop(0)

        self.x, self.y = float(x), float(y)
        return xs, ys, zs

    def get_history(self):
        return (np.asarray(self.history_x, np.float32),
                np.asarray(self.history_y, np.float32))

    
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
    
    def step_batch(self, frames):
        sigma, rho, beta, dt = self.sigma, self.rho, self.beta, self.dt
        x = float(self.x); y = float(self.y); z = float(self.z)

        xs = np.empty(frames, dtype=np.float32)
        ys = np.empty(frames, dtype=np.float32)
        zs = np.empty(frames, dtype=np.float32)

        for i in range(frames):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt

            xs[i] = x; ys[i] = y; zs[i] = z
            self.history.append((float(x), float(y), float(z)))
            while len(self.history) > 500:
                self.history.pop(0)

        self.x, self.y, self.z = float(x), float(y), float(z)
        return xs, ys, zs

    def get_history(self):
        if not self.history:
            return (np.array([], np.float32), np.array([], np.float32), np.array([], np.float32))
        hx, hy, hz = zip(*self.history[-2000:])
        return (np.asarray(hx, np.float32), np.asarray(hy, np.float32), np.asarray(hz, np.float32))

    
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
    
    def step_batch(self, frames):
        '''Manual RK4 integrator across a block; GPU-friendly and avoids odeint in the callback loop XD'''

        xp = _xp
        dt = 1.0/44100.0 # tie to audio rate
        x = float(self.initial_state[0])
        v = float(self.initial_state[1])
        t = float(self.t)
        xs = xp.empty(frames, dtype=xp.float32)
        vs = xp.empty(frames, dtype=xp.float32)

        for i in range(frames):
            # defining Duffing deriviaties
            def f_x(x, v, t): return v
            def f_v(x, v, t): return -self.delta * v - self.alpha * x - self.beta * x**3 + self.gamma * np.cos(self.omega * t)

            k1x = f_x(x, v, t);         k1v = f_v(x, v, t)
            k2x = f_x(x + 0.5*dt*k1x, v + 0.5*dt*k1v, t + 0.5*dt);  k2v = f_v(x + 0.5*dt*k1x, v + 0.5*dt*k1v, t + 0.5*dt)
            k3x = f_x(x + 0.5*dt*k2x, v + 0.5*dt*k2v, t + 0.5*dt);  k3v = f_v(x + 0.5*dt*k2x, v + 0.5*dt*k2v, t + 0.5*dt)
            k4x = f_x(x + dt*k3x, v + dt*k3v, t + dt);              k4v = f_v(x + dt*k3x, v + dt*k3v, t + dt)
            x += (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            v += (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            t += dt
            xs[i] = x; vs[i] = v
            
            self.history.append((float(x), float(v), float(t)))
            if len(self.history) > 500:
                self.history.pop(0)
            self.initial_state = [float(x), float(v)]
            self.t = float(t)
            z = xp.linspace(-1.0, 1.0, frames, dtype=xp.float32)
            return _to_numpy(xs), _to_numpy(vs), _to_numpy(z)

    def get_history(self):
        if not self.history:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )
        hx, hv, ht = zip(*self.history[-2000:])
        return (
            np.asarray(hx, dtype=np.float32),
            np.asarray(hv, dtype=np.float32),
            np.asarray(ht, dtype=np.float32),
        )
    
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

