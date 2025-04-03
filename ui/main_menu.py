import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Hockey Menu")
        self.setGeometry(500, 100, 1200, 900)
        self.setStyleSheet("background-color: #1e1e1e;")

        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        button_definitions = {
            "Manual AI vs AI": self.run_manual_vs_ai,
            "Touchscreen test vs AI": self.touchscreen_vs_ai,
            "Ikke enda klar 2": self.placeholder,
            "Ikke enda klar 3": self.placeholder
        }

        for label, callback in button_definitions.items():
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(callback)

            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px;
                    padding: 15px;
                    color: white;
                    background-color: #2c2c2c;
                    border: 2px solid red;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #3d3d3d;
                }
            """)

            layout.addWidget(btn)

    def run_manual_vs_ai(self):
        script_path = os.path.abspath("test_scripts/main_copy_for_ui_testing_ai_vs_ai.py")
        subprocess.Popen([sys.executable, script_path])

    def touchscreen_vs_ai(self):
        script_path = os.path.abspath("../environment/main_copy_for_ui_testing_touchscreen.py")
        subprocess.Popen([sys.executable, script_path])

    def placeholder(self):
        print("placeholder for no, bytter til andre AI modes senere")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec_())
