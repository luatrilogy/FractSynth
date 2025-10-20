import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class SpectrumPlot(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)

    def update_plot(self, signal):
        signal = signal[-2048:]
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        if np.max(spectrum) > 0:
            spectrum /= np.max(spectrum)

        plt.figure(figsize=(4, 2))
        plt.plot(spectrum, color='blue')
        plt.axis('off')
        plt.tight_layout()

        tmp_path = os.path.join(tempfile.gettempdir(), "spectrum.png")
        plt.savefig(tmp_path, dpi=100)
        plt.close()
        self.setPixmap(QPixmap(tmp_path))

class ChaosVisualizer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_history(self, history):
        try:
            history = [np.array(h) for h in history]
            if not all(np.isfinite(h).all() for h in history):
                return  # skip invalid frames

            self.ax.clear()
            if len(history) == 3:
                self.ax.plot3D(*history, lw=0.5, color='blue')
            elif len(history) == 2:
                self.ax.plot(*history, lw=0.5, color='blue')
            else:
                self.ax.plot(history[0], lw=0.5, color='blue')
            self.canvas.draw()
        except Exception as e:
            print("Error in chaos visualizer:", e)

    def update_visual(self, engine):
        fig = plt.figure(figsize=(6.5, 6.5))
        try:
            history = list(engine.get_history())
            if len(history) == 3:
                x, y, z = history
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(x, y, z, color='blue', linewidth=0.5)
            elif len(history) == 2:
                x, v = history
                ax = fig.add_subplot(111)
                ax.plot(x, v, 'o', markersize=1,)
            else:
                x = history[0]
                ax = fig.add_subplot(111)
                ax.plot(x, 'r')
            ax.axis('off')
            tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0)
            self.setPixmap(QPixmap(tmpfile.name))
            tmpfile.close()
        except Exception as e:
            print("Visualizer error:", e)
        finally:
            plt.close(fig)

    def smooth_series(data, alpha=0.2):
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return np.array(smoothed)