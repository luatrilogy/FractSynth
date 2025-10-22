import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt

class SpectrumPlot(QLabel):  # keeping the same name to avoid refactors
    """
    Fast time-domain waveform renderer (mono mix of stereo).
    Draws directly with QPainter (no Matplotlib, no temp files).
    """
    def __init__(self, parent=None, window=4096):
        super().__init__(parent)
        self._win = int(window)               # number of recent samples to show
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setScaledContents(True)
        self._last_pix = None

    def update_plot(self, signal):
        if signal is None:
            return
        x = np.asarray(signal)

        # downmix stereo -> mono
        if x.ndim == 2:
            x = x.mean(axis=1)

        n = x.size
        if n == 0:
            return

        # take the most recent window; pad if shorter
        if n < self._win:
            x = np.pad(x, (self._win - n, 0), mode="constant")
        else:
            x = x[-self._win:]

        # normalize to [-1, 1] with a tiny floor to avoid division by ~0
        peak = float(np.max(np.abs(x)))
        if peak < 1e-9:
            x = np.zeros_like(x)
        else:
            x = x / peak

        # prepare image to draw
        w = max(300, self.width() or 300)
        h = max(120, self.height() or 120)
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.black)

        painter = QPainter(img)

        # draw midline
        mid_y = h // 2
        mid_pen = QPen(Qt.darkGray)
        mid_pen.setWidth(1)
        painter.setPen(mid_pen)
        painter.drawLine(0, mid_y, w - 1, mid_y)

        # draw waveform
        pen = QPen(Qt.white)
        pen.setWidth(1)
        painter.setPen(pen)

        # map samples -> screen coords
        xs = np.linspace(0, w - 1, x.size).astype(np.int32)
        # y: +1 at top? we want +1 to be top-negative; invert amplitude
        ys = (mid_y - (x * (h * 0.45))).astype(np.int32)  # 0.9 of full height

        # polyline
        for i in range(x.size - 1):
            painter.drawLine(int(xs[i]), int(ys[i]), int(xs[i + 1]), int(ys[i + 1]))

        painter.end()
        self._last_pix = QPixmap.fromImage(img)
        self.setPixmap(self._last_pix)

class ChaosVisualizer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(6.5, 6.5), dpi=100)
        self._ax = self._fig.add_subplot(111, projection=None)
        self._canvas = FigureCanvas(self._fig)

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
        hist = list(engine.get_history())
        self._fig.clf()

        if getattr(engine, "type", "").upper() == "LOGISTIC":
            x = hist[0]
            self._ax = self._fig.add_subplot(111)
            self._ax.plot(x, 'r.', markersize=2)
        elif len(hist) == 3:
            x, y, z = hist
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._ax.plot(x, y, z, linewidth=0.5)
        elif len(hist) == 2:
            x, v = hist
            self._ax = self._fig.add_subplot(111)
            self._ax.plot(x, v, 'o', markersize=1)
        else:
            x = hist[0]
            self._ax = self._fig.add_subplot(111)
            self._ax.plot(x, linewidth=0.7)
        self._ax.axis('off')

        self._canvas.draw()
        w, h = self._canvas.get_width_height()
        buf = np.frombuffer(self._canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        qimg = QImage(buf.data, w, h, 3*w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def smooth_series(data, alpha=0.2):
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return np.array(smoothed)
    
    def _draw_logistic_cobweb(self, painter, w, h, x_hist, r, steps=200):
        # axes background
        painter.fillRect(0, 0, w, h, Qt.black)

        # map [0,1] -> pixels
        def X(u): return int(u * (w - 1))
        def Y(v): return int((1.0 - v) * (h - 1))  # y up

        # draw f(x)=r x (1-x) curve
        pen_curve = QPen(Qt.gray); pen_curve.setWidth(1)
        painter.setPen(pen_curve)
        prev = None
        N = 600
        for i in range(N):
            u = i / (N - 1)
            v = r * u * (1.0 - u)
            p = (X(u), Y(v))
            if prev is not None:
                painter.drawLine(prev[0], prev[1], p[0], p[1])
            prev = p

        # draw y=x line
        pen_diag = QPen(Qt.darkGray); pen_diag.setWidth(1)
        painter.setPen(pen_diag)
        painter.drawLine(X(0), Y(0), X(1), Y(1))

        # cobweb from the last point
        if x_hist is None or len(x_hist) < 2:
            return
        # take last 'steps' points after burn-in
        xs = x_hist[-(steps+1):].astype(float).clip(0, 1)

        pen_web = QPen(Qt.white); pen_web.setWidth(1); pen_web.setColor(QColor(255,255,255,140))
        painter.setPen(pen_web)

        x = xs[0]
        for i in range(steps):
            # vertical: (x, x) -> (x, f(x))
            fx = r * x * (1.0 - x)
            painter.drawLine(X(x), Y(x), X(x), Y(fx))
            # horizontal: (x, f(x)) -> (f(x), f(x))
            painter.drawLine(X(x), Y(fx), X(fx), Y(fx))
            x = fx