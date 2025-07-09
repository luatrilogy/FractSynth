from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen

# MIDI notes from C4 (60) to C5 (72)
WHITE_KEYS = [
    ('C', 60), ('D', 62), ('E', 64),
    ('F', 65), ('G', 67), ('A', 69), ('B', 71), ('C', 72)
]
BLACK_KEYS = [
    ('C#', 61, 0), ('D#', 63, 1),        # Between C-D, D-E
    ('F#', 66, 3), ('G#', 68, 4), ('A#', 70, 5)  # Between F-G, G-A, A-B
]

class PianoWidget(QWidget):
    def __init__(self, on_note_callback):
        super().__init__()
        self.on_note = on_note_callback
        self.white_key_width = 40
        self.black_key_width = 28
        self.black_key_height = 60
        self.white_keys = len(WHITE_KEYS)
        self.setFixedHeight(100)
        self.setMinimumWidth(self.white_key_width * self.white_keys)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1))

        # Draw white keys
        for i, (note, midi) in enumerate(WHITE_KEYS):
            rect = QRectF(i * self.white_key_width, 0, self.white_key_width, self.height())
            painter.setBrush(QColor(255, 255, 255))
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignBottom | Qt.AlignHCenter, note)

        # Draw black keys on top
        for note, midi, pos in BLACK_KEYS:
            x = (pos + 1) * self.white_key_width - self.black_key_width / 2
            rect = QRectF(x, 0, self.black_key_width, self.black_key_height)
            painter.setBrush(QColor(0, 0, 0))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()

        # First, check black keys
        for note, midi, pos in BLACK_KEYS:
            bx = (pos + 1) * self.white_key_width - self.black_key_width / 2
            rect = QRectF(bx, 0, self.black_key_width, self.black_key_height)
            if rect.contains(x, y):
                return self.trigger_note(midi)

        # Then check white keys
        white_index = int(x / self.white_key_width)
        if 0 <= white_index < len(WHITE_KEYS):
            return self.trigger_note(WHITE_KEYS[white_index][1])

    def trigger_note(self, midi_note):
        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        self.on_note(freq)