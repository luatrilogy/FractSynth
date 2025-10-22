import numpy as np
import sounddevice as sd 
import threading
import queue

_HAS_CUPY = False
xp = np

try:
    import cupy as _cp
    try:
        # Only enable if at least one CUDA device is usable
        if _cp.cuda.runtime.getDeviceCount() > 0:
            xp = _cp
            _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
except Exception:
    _HAS_CUPY = False

from chaos import LogisticMap, HenonMap, LorenzAttractor #, DuffingOscillator

def to_xp(a):
    return xp.asarray(a)

def to_np(a):
    if 'cupy' in xp.__name__:
        import cupy as cp
        return cp.asnumpy(a)
    return np.asarray(a)

def _onepole_block(x, z1, alpha):
    """Sample-by-sample one-pole smoothing on the current backend (xp)."""
    y = xp.empty_like(x, dtype=xp.float32)
    p = xp.asarray(z1, dtype=xp.float32)
    for i in range(x.size):
        p = p + alpha * (x[i] - p)
        y[i] = p
    # return the block and the final state for next call
    return y, float(p)

class FractSynth:
    def __init__(self, sample_rate=44100, num_harmonics=8):
        # --- Core synth settings (same as before)
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.freq_base = 220.0  # A3
        self.phase = [0.0 for _ in range(self.num_harmonics)]
        self.chaos_type = "Logisitc"
        self.r = 3.9
        self.gain = 0.01
        self.playing = False
        self.engines = self._init_engines()
        self.chaos_morph = 0.5
        self.attractor = None

        # --- Signal chain (same order you had)
        self.lpf = LowPassFilter(cuttoff=3000)
        self.env = ADSREnvelope(self.sample_rate)
        self.delay1 = Delay(delay_time=0.15, feedback=0.3)
        self.delay2 = Delay(delay_time=0.4, feedback=0.2)

        # Optionally start opened; leave commented if you prefer silence until a note
        # self.env.note_on()

        # --- Runtime state
        self.fade_out = False
        self.last_gain = self.gain
        self.time = 0.0
        self.velocity = 1.0

        # ======================================================================
        # Background AUDIO PRODUCER (precomputes audio blocks off the callback)
        # ======================================================================
        # This must match the stream blocksize in synth_gui.py (currently 512) :contentReference[oaicite:1]{index=1}
        self.frames = 512
        # Small queue for ~8 blocks of headroom (~93 ms at 44.1kHz & 512 frames)
        self.q = queue.Queue(maxsize=8)

        # Run a lightweight producer that continuously fills the queue
        def _producer():
            while True:
                # Generate next audio block; if env is 'off' this is silence (that’s ok)
                block = self.generate(self.frames)
                try:
                    self.q.put(block, timeout=0.25)
                except queue.Full:
                    # If GUI or callback hiccups, drop the oldest block silently
                    try:
                        _ = self.q.get_nowait()
                    except queue.Empty:
                        pass
                    # Try again to avoid starving the callback
                    try:
                        self.q.put(block, timeout=0.05)
                    except queue.Full:
                        pass

        self.audio_thread = threading.Thread(target=_producer, daemon=True)
        self.audio_thread.start()
    
        self.last_block = np.zeros((512, 2), dtype=np.float32)
    
    def _init_engines(self):
        engines = []

        for i in range(self.num_harmonics):
            if self.chaos_type == "Logisitc":
                engines.append(LogisticMap(self.r, 0.5 + i * 0.01))
            elif self.chaos_type == "Henon":
                engines.append(HenonMap())
            elif self.chaos_type == "Lorenz":
                engines.append(LorenzAttractor())
            elif self.chaos_type == "Duffing":
                engines.append(DuffingOscillator())
                
        return engines
    
    def set_r(self, r):
        self.r = r
        
        if self.chaos_type == "Logistic":
            # Update r for existing engines or rebuild them
            for i in range(len(self.engines)):
                self.engines[i] = LogisticMap(r, 0.5 + i * 0.01)

    def set_gain(self, gain):
        self.gain = 0.9 * self.gain + 0.1 * gain

    def set_freq(self, freq): 
        self.freq_base = freq

    def set_type(self, chaos_type):
        self.chaos_type = chaos_type
        self.engines = self._init_engines()
    
    def generate(self, frames):
        # Allocate device (xp) buffers
        outL = xp.zeros(frames, dtype=xp.float32)
        outR = xp.zeros(frames, dtype=xp.float32)

        for i, engine in enumerate(self.engines):
            self.attractor = engine

            # --- Morph parameters by chaos_morph (kept lightweight)
            m = float(self.chaos_morph)
            etype = getattr(engine, "type", "")

            if etype == "LOGISITC":
                # sweep r from stable-ish to fully chaotic, clamp to [0,4]
                stable_r, chaotic_r = 3.2, 3.999
                r = stable_r * (1.0 - m) + chaotic_r * m
                r = max(0.0, min(4.0, r))
                if hasattr(engine, "r"):
                    engine.r = r

            elif etype == "HENON":
                # sweep into/within chaotic regime (a,b)
                a0, a1 = 1.2, 1.4
                b0, b1 = 0.2, 0.3
                engine.a = a0 * (1.0 - m) + a1 * m
                engine.b = b0 * (1.0 - m) + b1 * m

            elif etype == "LORENZ":
                # sweep rho/sigma to move between regimes
                rho0, rho1 = 18.0, 35.0
                sig0, sig1 = 8.0, 12.0
                engine.rho = rho0 * (1.0 - m) + rho1 * m
                engine.sigma = sig0 * (1.0 - m) + sig1 * m
                # keep beta near canonical value

            # Optional engine modulation hook
            t_now = self.time + i * frames / self.sample_rate
            if hasattr(engine, "modulate_params"):
                engine.modulate_params(t_now, velocity=self.velocity)

            # --- Get a block of attractor states
            if hasattr(engine, "step_batch"):
                x_vals, y_vals, z_vals = engine.step_batch(frames)
            else:
                # Fallback: per-sample stepping (slower)
                x_vals = np.zeros(frames, dtype=np.float32)
                y_vals = np.zeros(frames, dtype=np.float32)
                z_vals = np.zeros(frames, dtype=np.float32)
                for j in range(frames):
                    x, y, z = engine.step()
                    x_vals[j], y_vals[j], z_vals[j] = x, y, z

            xb = to_xp(x_vals).astype(xp.float32, copy=False)
            yb = to_xp(y_vals).astype(xp.float32, copy=False)
            zb = to_xp(z_vals).astype(xp.float32, copy=False)

            xb = xp.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = xp.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            zb = xp.nan_to_num(zb, nan=0.0, posinf=0.0, neginf=0.0)

            # normalization (preserve relative chaos shape)
            def norm_pm1(a):
                a = a - a.mean()
                s = a.std()
                s = s if s > 1e-6 else 1.0
                return np.clip(a / (3.0*s), -1.0, 1.0)

            x_n = norm_pm1(xb)
            y_n = norm_pm1(yb)
            z_n = norm_pm1(zb)

            # --- Morph-dependent modulation depths ---
            m = float(self.chaos_morph)  # 0..1
            freq_depth = 0.05 + 0.95 * m   # pitch movement
            amp_depth  = 0.10 + 0.90 * m   # loudness modulation
            pan_depth  = 0.20 + 0.80 * m   # stereo spread

            # Frequency modulation centered on the clicked note
            freq = self.freq_base * (1.0 + freq_depth * x_n)

            # Amplitude and stereo control
            amp = xp.clip(0.15 + amp_depth * (y_n * 0.5 + 0.5), 0.0, 1.0)
            pan = xp.clip(pan_depth * z_n, -1.0, 1.0)

            phase_inc = (2.0 * xp.pi * freq) / self.sample_rate
            phase0 = xp.asarray(self.phase[i], dtype=xp.float32)
            phase  = phase0 + xp.cumsum(phase_inc)
            wave   = amp * xp.sin(phase)
            self.phase[i] = float((to_np(phase[-1]) % (2.0 * np.pi)))

            shaped = xp.tanh(wave * 0.8)
            L = to_xp((1.0 - pan) * 0.5)   # <— ensure xp
            R = to_xp((1.0 + pan) * 0.5)
            outL += shaped * L
            outR += shaped * R

        env = to_xp(self.env.generate(frames)).astype(xp.float32, copy=False)
        outL *= env
        outR *= env

        self.last_gain = 0.95 * self.last_gain + 0.05 * self.gain
        outL *= self.last_gain
        outR *= self.last_gain

        # Convert ONCE to NumPy for sounddevice and CPU effects
        out_np = np.stack([to_np(outL), to_np(outR)], axis=1).astype(np.float32)

        # CPU-only processing is fine here
        out_np[:, 0] = self.lpf.process(out_np[:, 0])
        out_np[:, 1] = self.lpf.process(out_np[:, 1])

        # Optional fade-out of a single block
        if getattr(self, "fade_out", False):
            fade = np.linspace(1.0, 0.0, frames, dtype=np.float32)
            out_np *= fade[:, None]
            self.fade_out = False

        # Advance global time
        self.time += frames / self.sample_rate

        # Return as float32; no extra *self.gain here (already baked into last_gain)
        return out_np.astype(np.float32)   
     
    def audio_callback(self, outdata, frames, time, status):
        try:
            block = self.q.get_nowait()
            self.last_block = block

            if block.shape[0] != frames:
                # pad or trim to the requested size
                if block.shape[0] < frames:
                    tmp = np.zeros((frames, 2), dtype=np.float32)
                    tmp[:block.shape[0]] = block
                    outdata[:] = tmp
                else:
                    outdata[:] = block[:frames]
            else:
                outdata[:] = block
        except queue.Empty:
            outdata.fill(0)

    def set_filter_cutoff(self, cutoff_hz):
        self.lpf.cutoff = cutoff_hz
        self.lpf.alpha = self.lpf.compute_alpha()

    def set_velocity(self, velocity):
        self.velocity = max(0.0, min(velocity, 1.0))

    def set_blocksize(self, frames: int):
        self.frames = int(frames)
        self.last_block = np.zeros((self.frames, 2), dtype=np.float32)  # keep in sync
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def flush_queue(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

class LowPassFilter:
    def __init__(self, cuttoff=3000, sample_rate=44100):
        self.cutoff = cuttoff
        self.sample_rate = sample_rate
        self.alpha = self.compute_alpha()
        self.prev = 0.0

    def compute_alpha(self):
        rc = 1 / (2 * np.pi * self.cutoff)
        dt = 1 / self.sample_rate
        return dt / (rc + dt)
    
    def process(self, x):
        y = np.zeros_like(x)
        for i in range(len(x)):
            self.prev = self.prev + self.alpha * (x[i] - self.prev)
            y[i] = self.prev
        return y
    
class ADSREnvelope:
    def __init__(self, sample_rate, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
        self.sample_rate = sample_rate
        self.attack = int(attack * sample_rate)
        self.decay = int(decay * sample_rate)
        self.release = int(release * sample_rate)
        self.sustain = sustain
        self.state = 'off'
        self.counter = 0
        self.env_value = 0.0
        self.release_start = 0.0

    def note_on(self):
        self.state = 'attack'
        self.counter = 0

    def note_off(self):
        self.state = 'release'
        self.counter = 0
        self.release_start = self.env_value
    
    def generate(self, frames):
        env = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            if self.state == 'attack':
                if self.counter < self.attack:
                    self.env_value = self.counter / max(1, self.attack)
                else:
                    self.state = 'decay'
                    self.counter = 0
            elif self.state == 'decay':
                if self.counter < self.decay:
                    self.env_value = 1 - (1 - self.sustain) * (self.counter / self.decay)
                else:
                    self.state = 'sustain'
            elif self.state == 'sustain':
                self.env_value = self.sustain
            elif self.state == 'release':
                if self.counter < self.release:
                    self.env_value = self.release_start * (1 - self.counter / self.release)
                else:
                    self.env_value = 0.0
                    self.state = 'off'

            env[i] = self.env_value
            self.counter += 1

        return env 
    
class Delay:
    def __init__(self, delay_time=0.3, feedback=0.3, sample_rate=44100):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * delay_time)
        self.buffer = np.zeros((self.buffer_size, 2))
        self.feedback = feedback
        self.index = 0

    def process(self, input_sample):
        delayed_sample = self.buffer[self.index, :]
        self.buffer[self.index, :] = input_sample + delayed_sample * self.feedback
        self.index = (self.index + 1) % self.buffer_size
        return delayed_sample