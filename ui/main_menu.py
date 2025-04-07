import sys
import subprocess
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QLabel, QDialog, QGridLayout
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

class StatsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Match Statistics")
        self.setGeometry(600, 200, 400, 300)

        layout = QGridLayout()
        layout.addWidget(QLabel("AI Wins: 12"), 0, 0)
        layout.addWidget(QLabel("Player Wins: 8"), 1, 0)
        layout.addWidget(QLabel("AI Win Rate: 60%"), 2, 0)
        layout.addWidget(QLabel("Avg Puck Speed: 1.2 m/s"), 3, 0)

        self.setLayout(layout)

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

        banner = QLabel()
        banner.setAlignment(Qt.AlignCenter)
        banner.setText("Air Hockey Simulator")
        banner.setStyleSheet("color: white;")
        banner.setFont(QFont("Arial", 36, QFont.Bold))
        layout.addWidget(banner)

        button_definitions = {
            "Manual AI vs AI": self.run_manual_vs_ai,
            "Touchscreen test vs AI": self.touchscreen_vs_ai,
            "Show Stats": self.show_stats,
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
        script_path = os.path.abspath("test_scripts/main_copy_for_ui_testing_touchscreen.py")
        subprocess.Popen([sys.executable, script_path])

    def show_stats(self):
        self.stats_window = StatsWindow()
        self.stats_window.exec_()

    def placeholder(self):
        print("placeholder for now")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec_())
