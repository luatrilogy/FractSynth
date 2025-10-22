import threading
import sounddevice as sd

from piano import PianoWidget
from PyQt5.QtCore import Qt, QTimer
from synth_engine import FractSynth
from matplotlib.figure import Figure
from lyapunov_calc import calculate_lyapunov
from visuals import SpectrumPlot, ChaosVisualizer
from presets import presets, save_presets, load_presets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout, QPushButton, QComboBox, QHBoxLayout, QDesktopWidget

class FractSynthGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FractSynth")
        
        screen = QDesktopWidget().availableGeometry()
        max_width = screen.width()
        max_height = screen.height()
        self.resize(min(1200, max_width), min(900, max_height))

        self.le_history = []
        self.le_time = []
        self.old_le = 0

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Consolas;
            }
            QSlider::groove:horizontal {
                background: #444;
                height: 8px;
            }
            QSlider::handle:horizontal {
                background: #00ffff;
                border: 1px solid #5c5c5c;
                width: 14px;
            }
            QComboBox, QPushButton {
                background-color: #333;
                border: 1px solid #888;
                padding: 4px;
            }
        """)

        self.synth = FractSynth()
        self.plot = SpectrumPlot()
        self.piano = PianoWidget(self.handle_piano_note)

        self.chaos_display = ChaosVisualizer(self)
        self.chaos_display.setMinimumHeight(200)

        self.presets = presets
        self.presets.update(load_presets())

        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # CHAOS ALGORITHM
        self.chaos_type_box = QComboBox()
        for ctype in ["Logisitc", "Henon", "Lorenz"]: # "Duffing"
            self.chaos_type_box.addItem(ctype)
        self.chaos_type_box.currentTextChanged.connect(self.change_type)
        layout.addWidget(QLabel("Chaos Type"))
        layout.addWidget(self.chaos_type_box)

        # BLOCKSIZE SLIDER
        BLOCKSIZE = 512 #6144 
        self.current_blocksize = BLOCKSIZE

        self.blocksize_label = QLabel("Blocksize - ")
        self.blocksize_slider = QSlider(Qt.Horizontal)
        self.blocksize_slider.setMinimum(128)
        self.blocksize_slider.setMaximum(16384)
        self.blocksize_slider.setValue(8192)
        self.blocksize_slider.setSingleStep(32)
        self.blocksize_slider.setPageStep(128)
        self.blocksize_slider.valueChanged.connect(self.update_blocksize_from_slider)

        layout.addWidget(QLabel("Blocksize (frames)"))
        layout.addWidget(self.blocksize_slider)
        layout.addWidget(self.blocksize_label)

        self.blocksize_label.setText(f"Blocksize: {BLOCKSIZE} frames")

        # R SLIDER
        self.chaos_morph_slider = QSlider(Qt.Horizontal)
        self.chaos_morph_slider.setMinimum(0)
        self.chaos_morph_slider.setMaximum(100)
        self.chaos_morph_slider.setValue(50)  # 50%
        self.chaos_morph_slider.valueChanged.connect(self.update_chaos_morph)
        layout.addWidget(QLabel("Chaos Morph"))
        layout.addWidget(self.chaos_morph_slider)

        #self.r_slider = QSlider( Qt.Horizontal )
        #self.r_slider.setMinimum(300)
        #self.r_slider.setMaximum(400)
        #self.r_slider.setValue(390)
        #self.r_slider.valueChanged.connect(self.update_r)
        #layout.addWidget(QLabel("Chaos R Value"))
        #layout.addWidget(self.r_slider)

        # GAIN SLIDER
        self.gain_slider = QSlider( Qt.Horizontal )
        self.gain_slider.setMinimum(0)
        self.gain_slider.setMaximum(100)
        self.gain_slider.setValue(5)
        self.gain_slider.valueChanged.connect(self.update_gain)
        layout.addWidget(QLabel("Gain"))
        layout.addWidget(self.gain_slider)

        # FREQUENCY SLIDER
        self.freq_slider = QSlider( Qt.Horizontal )
        self.freq_slider.setMinimum(100)
        self.freq_slider.setMaximum(800)
        self.freq_slider.setValue(220)
        self.freq_slider.valueChanged.connect(self.update_freq)

        # LOW PASS FILTER SLIDER
        self.tone_slider = QSlider(Qt.Horizontal)
        self.tone_slider.setMinimum(100)
        self.tone_slider.setMaximum(5000)
        self.tone_slider.setValue(2000)
        self.synth.set_filter_cutoff(2000)
        self.tone_slider.valueChanged.connect(self.update_filter_cutoff)
        layout.addWidget(QLabel("Tone (Low-Pass Filter Cutoff)"))
        layout.addWidget(self.tone_slider)
        
        # ADSR SLIDER
        # ATTACK
        self.attack_slider = QSlider(Qt.Horizontal)
        self.attack_slider.setMinimum(1)
        self.attack_slider.setMaximum(2000)
        self.attack_slider.setValue(1000) # default to 1 sec
        self.attack_slider.valueChanged.connect(self.update_attack)
        layout.addWidget(QLabel("Attack (ms)"))
        layout.addWidget(self.attack_slider)

        #DECAY
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setMinimum(1)
        self.decay_slider.setMaximum(2000)
        self.decay_slider.setValue(300)
        self.decay_slider.valueChanged.connect(self.update_decay)
        layout.addWidget(QLabel("Decay (ms)"))
        layout.addWidget(self.decay_slider)

        # SUSTAIN (0.0 to 1.0 scaled as 0 to 100)
        self.sustain_slider = QSlider(Qt.Horizontal)
        self.sustain_slider.setMinimum(0)
        self.sustain_slider.setMaximum(100)
        self.sustain_slider.setValue(70)
        self.sustain_slider.valueChanged.connect(self.update_sustain)
        layout.addWidget(QLabel("Sustain (0–1)"))
        layout.addWidget(self.sustain_slider)

        #RELEASE
        self.release_slider = QSlider(Qt.Horizontal)
        self.release_slider.setMinimum(1)
        self.release_slider.setMaximum(2000)
        self.release_slider.setValue(500)
        self.release_slider.valueChanged.connect(self.update_release)
        layout.addWidget(QLabel("Release (ms)"))
        layout.addWidget(self.release_slider)

        # PRESETS
        self.preset_box = QComboBox()
        for name in self.presets:
            self.preset_box.addItem(name)
        self.preset_box.currentTextChanged.connect(self.load_presets)
        
        self.save_button = QPushButton("Save Presets")
        self.save_button.clicked.connect(lambda: save_presets())

        self.toggle_button = QPushButton("Play")
        self.toggle_button.clicked.connect(self.toggle_playback)

        layout.addWidget(QLabel("Presets"))
        layout.addWidget(self.preset_box)
        layout.addWidget(self.save_button)
        layout.addWidget(self.toggle_button)
        
        layout.addWidget(QLabel("Spectral View"))
        layout.addWidget(self.plot)

        right_layout.addWidget(self.chaos_display)

        #self.figure = Figure()
        #self.ax = self.figure.add_subplot(111)
        #self.canvas = FigureCanvas(self.figure)

        #right_layout.addWidget(self.canvas)
        
        # LE PLOT OVERTIME
        self.le_timer = QTimer()
        self.le_timer.timeout.connect(self.update_lyapunov_label)
        self.le_timer.start(2000)
        self.le_timer.start()
        
        layout.addWidget(QLabel("Virtual Piano"))
        layout.addWidget(self.piano)
        
        self.luanpunov_calc = QLabel("Lyapunov Exponent: 0.0")
        self.luanpunov_calc.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.luanpunov_calc)

        main_layout.addLayout(layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # keep synth + stream in sync
        self.synth.set_blocksize(BLOCKSIZE)
        
        self.stream = sd.OutputStream(
            callback=self.synth.audio_callback,
            samplerate=44100,
            channels=2,
            dtype='float32',
            blocksize=BLOCKSIZE,          # You can raise this to 1024 if skips persist
            latency='high',         # or latency=0.06 for ~60ms buffer
            dither_off=True
        )

        self.stream.start()

        self._vis_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visuals)
        self.timer.start(150)  # 10 FPS base

    def update_r(self, value):
        self.synth.set_r(value / 100)
    
    def update_gain(self, value):
        self.synth.set_gain(value/100)

    def update_freq(self, value):
        self.synth.set_freq(value)

    def change_type(self, name):
        self.synth.set_type(name)

    def load_presets(self, name):
        p = self.presets[name]
        #self.r_slider.setValue(int(p['r'] * 100))
        self.gain_slider.setValue(int(p['gain'] * 100))
        self.freq_slider.setValue(int(p['base_freq']))
        self.chaos_type_box.setCurrentText(p.get("type", "Logistic"))

    def update_visuals(self):
        self._vis_counter = (self._vis_counter + 1) % 3  # ~3–4 FPS
        if self._vis_counter != 0:
            return

        if self.synth.playing and getattr(self.synth, "last_block", None) is not None:
            self.plot.update_plot(self.synth.last_block)
        if self.synth.engines and hasattr(self.synth.engines[0], 'get_history'):
            self.chaos_display.update_visual(self.synth.engines[0])          

    def handle_piano_note(self, freq):
        self.synth.set_freq(freq)
        self.synth.env.note_off()
        self.synth.flush_queue()
        self.synth.env.note_on()
        self.synth.playing = True
        self.toggle_button.setText("Stop")
 
    def toggle_playback(self):
        if self.synth.playing:
            self.synth.env.note_off()
            self.synth.playing = False
            self.toggle_button.setText("Play")
        else:
            # Fade out gracefully instead of hard stop
            self.synth.fade_out = True
    
    def update_filter_cutoff(self, value):
        self.synth.set_filter_cutoff(value)

    # ADSR SLIDERs
    def update_attack(self, value):
        self.synth.env.attack = int(value * self.synth.sample_rate / 1000)

    def update_decay(self, value):
        self.synth.env.decay = int(value * self.synth.sample_rate / 1000)

    def update_sustain(self, value):
        self.synth.env.sustain = value / 100.0

    def update_release(self, value):
        self.synth.env.release = int(value * self.synth.sample_rate / 1000)

    # CHAOS MORPH
    def update_chaos_morph(self, value: int):
        # map 0..100 slider to 0.0..1.0
        self.synth.chaos_morph = value / 100.0

    # LYAPUNOV CALC
    def update_lyapunov_label(self):
        if hasattr(self.synth, 'attractor') and self.synth.attractor is not None:
            if hasattr(self.synth.attractor, 'lyapunov_exponent'):
                func = self.synth.attractor.get_map_function()
                le, _ = calculate_lyapunov(self.synth.attractor, func, steps=50, dt=0.01)

                if le > 0:
                    self.old_le = le
            else:
                le = None

            if le is not None:
                expected = self.synth.attractor.get_expected_le_range()

                if isinstance(expected, tuple) and len(expected) == 2:
                    a, b = expected
                else:
                    a, b = 0.0, 0.0

                if le < 0:
                    le = self.old_le

                self.luanpunov_calc.setText(f"Lyapunov: {le:.5f} (Expected: {a:.2f}–{b:.2f})")
                self.synth.attractor.lyapunov_exponent = le

    # LE PLOT OVER TIME
    def update_le_plot(self):
        pass
    '''
        if hasattr(self.synth, "attractor") and self.synth.attractor is not None:
            le = self.synth.attractor.lyapunov_exponent
            self.le_history.append(le)
            self.le_time.append(len(self.le_time))

            if len(self.le_history) > 100:
                self.le_history.pop(0)
                self.le_time.pop(0)

            self.ax.clear()
            self.ax.set_title("Lyapunov Exponent Over Time")
            self.ax.set_ylabel("LE")
            self.ax.set_ylim(0, 1.5)
            self.ax.set_xlabel("Update Step")

            if self.synth.chaos_type == "Logistic":
                x, y = zip(*self.synth.history)
                y_smoothed = self.smooth_series(y)
                self.ax.plot(x, y_smoothed, color='red', label="Lyanpunov Exponent")
            else:
                self.ax.plot(self.le_time, self.le_history, label="Lyanpunov Exponent")
            
            self.ax.legend()
            self.canvas.draw()
            '''
    
    # SMOOTH SERIRES
    def smooth_series(data, alpha=0.2):
        if len(data) == 0:
            return []
        smoothed = [data[0]]

        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return smoothed
    
    def update_blocksize_from_slider(self, value: int):
        if value != self.current_blocksize:
            self.recreate_stream(value)
    '''
    def _nearest_power_of_two(sef, n, min_pow=128, max_pow=16384):
        n = max(min_pow, min(int(n), max_pow))
        p = 1

        while p < n:
            p <<= 1

        lower = p >> 1
        if lower < min_pow: lower = min_pow
        if abs(n - lower) <= abs(p - n): return lower
        
        return p
    '''

    def recreate_stream(self, new_blocksize: int):
        try: 
            if hasattr(self, "stream") and self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        self.synth.set_blocksize(new_blocksize)
        self.current_blocksize = new_blocksize
        self.blocksize_label.setText(f"Blocksize: {new_blocksize} frames")

        self.stream = sd.OutputStream(
            callback=self.synth.audio_callback,
            samplerate=44100,
            channels=2,
            dtype='float32',
            blocksize=new_blocksize,
            latency='high',
            dither_off=True
        )

        self.stream.start()
