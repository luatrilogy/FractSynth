import numpy as np
import sounddevice as sd 

from chaos import LogisticMap, HenonMap, LorenzAttractor, DuffingOscillator

class FractSynth:
    def __init__(self, sample_rate=44100, num_harmonics=8):
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.freq_base = 220.0 # A3
        self.phase = [0.0 for _ in range(self.num_harmonics)]
        self.chaos_type = "Logisitc"
        self.r = 3.9
        self.gain = 0.01
        self.playing = False
        self.engines = self._init_engines() 
        self.chaos_morph = 0.5
        self.attractor = None
        
        self.lpf = LowPassFilter(cuttoff=3000)
        self.env = ADSREnvelope(self.sample_rate)
        self.delay1 = Delay(delay_time=0.15, feedback=0.3)
        self.delay2 = Delay(delay_time=0.4, feedback=0.2)

        self.fade_out = False
        self.last_gain = self.gain

        self.time = 0.0
        self.velocity = 1.0
    
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
        t = np.arange(frames) / self.sample_rate
        output = np.zeros((frames, 2), dtype=np.float32)

        for i in range(len(output)):
            dry = output[i]
            output[i] = dry + 0.5 * self.delay1.process(dry) + 0.3 * self.delay2.process(dry)

        for i, engine in enumerate(self.engines):
            self.attractor = engine

            if engine.type == "LOGISITC":
                stable_r = 3.2
                chaotic_r = 4.0

                self.r = stable_r * (1 - self.chaos_morph) + chaotic_r * self.chaos_morph
            elif engine.type == "DUFFING":
                stable_beta = 0.1
                chaotic_beta = 1.2

                engine.beta = stable_beta * (1 - self.chaos_morph) + chaotic_beta * self.chaos_morph

            t_now = self.time + i * frames / self.sample_rate

            if hasattr(engine, "modulate_params"):
                engine.modulate_params(t_now, velocity=self.velocity)

            x_vals = np.zeros(frames)
            y_vals = np.zeros(frames)
            z_vals = np.zeros(frames)

            #engine.r += 0.001 * np.sin(2 * np.pi * 0.2 * self.time)

            for j in range(frames):
                x, y, z = engine.step()  # audio-rate attractor updates
                x_vals[j], y_vals[j], z_vals[j] = x, y, z
            
            freq = np.interp(x_vals, [-2, 2], [100, 1000])         # pitch from x
            amp = np.interp(y_vals, [-2, 2], [0.1, 1.0])           # amplitude from y
            pan = np.interp(z_vals, [-2, 2], [-1.0, 1.0])          # panning from z

            phase_inc = 2 * np.pi * freq / self.sample_rate
            waveform = np.zeros(frames)
            phase = self.phase[i]

            for j in range(frames):
                waveform[j] = amp[j] * np.sin(phase)
                phase += phase_inc[j]

            self.phase[i] = phase % (2 * np.pi)

            #amp = np.array([engine.step() for _ in range(frames)])
            #freq = self.freq_base * (i + 1)
            #phase_increment = 2 * np.pi * freq / self.sample_rate
            #self.phase[i] += phase_increment * frames
            #attenuation = 1.0 / ((i + 1) ** 2)
            #wave = attenuation * amp * np.sin(2 * np.pi * freq * t + self.phase[i])

            #pan = 0.5 + 0.5 * np.tanh(z) 
            shaped_wave = np.tanh(waveform * 0.8)
            output[:, 0] += shaped_wave * (1 - pan) * 0.5
            output[:, 1] += shaped_wave * (pan) * 0.5

        envelope = self.env.generate(frames)
        output *= envelope[:, np.newaxis]

        self.last_gain = 0.95 * self.last_gain + 0.05 * self.gain
        output *= self.last_gain

        output = np.clip(output, -1.0, 1.0)
        output[:,0] = self.lpf.process(output[:,0])
        output[:,1] = self.lpf.process(output[:,1])

        if hasattr(self, "fade_out") and self.fade_out:
            fade = np.linspace(1, 0, frames)
            output[:,0] *= fade
            output[:,1] *= fade
            self.fade_out = False
        
        self.time += frames / self.sample_rate
        return (output * self.gain).astype(np.float32)
    
    def audio_callback(self, outdata, frames, time, status):
        outdata[:] = self.generate(frames)#.reshape(-1, 1)# if self.playing else np.zeros((frames, 2), dtype=np.float32)

    def set_filter_cutoff(self, cutoff_hz):
        self.lpf.cutoff = cutoff_hz
        self.lpf.alpha = self.lpf.compute_alpha()

    def set_velocity(self, velocity):
        self.velocity = max(0.0, min(velocity, 1.0))

class LowPassFilter:
    def __init__(self, cuttoff=3000, sample_rate=44100):
        self.cutoff = cuttoff
        self.sample_rate = sample_rate
        self.alpha = self.compute_alpha()
        self.prev = 0

    def compute_alpha(self):
        rc = 1 / (2 * np.pi * self.cutoff)
        dt = 1 / self.sample_rate
        return dt / (rc + dt)
    
    def process (self, x):
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
        env = np.zeros(frames)
        for i in range(frames):
            if self.state == 'attack':
                if self.counter < self.attack:
                    self.env_value = self.counter / self.attack
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
                    self.env_value = 0
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
    
        