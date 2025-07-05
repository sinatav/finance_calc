from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import sys
from interests import SimpleInterest

class FinanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finance Calculator")
        layout = QVBoxLayout()

        self.input_p = QLineEdit()
        self.input_r = QLineEdit()
        self.input_t = QLineEdit()
        button = QPushButton("Calculate Simple Interest")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Principal:"))
        layout.addWidget(self.input_p)
        layout.addWidget(QLabel("Rate (decimal):"))
        layout.addWidget(self.input_r)
        layout.addWidget(QLabel("Time:"))
        layout.addWidget(self.input_t)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            P = float(self.input_p.text())
            r = float(self.input_r.text())
            t = float(self.input_t.text())
            si = SimpleInterest(P, r, t)
            QMessageBox.information(self, "Result", f"Simple Interest: {si.calculate():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinanceApp()
    window.show()
    sys.exit(app.exec())
