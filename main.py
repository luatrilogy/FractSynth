import sys

from synth_gui import FractSynthGUI
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FractSynthGUI()
    window.show()
    sys.exit(app.exec_())