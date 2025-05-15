import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QPushButton, QLineEdit, QLabel, QFileDialog

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("AI Image Generator")
        self.setGeometry(100, 100, 1000, 800)

        # Center widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Dropdown menu
        self.option_menu = QComboBox(self)
        self.option_menu.addItems(["Option 1", "Option 2", "Option 3"])
        self.option_menu.setCurrentText("Select an option")
        layout.addWidget(self.option_menu)

        # Folder chooser button
        self.folder_path_label = QLabel("Selected Folder: None", self)
        self.folder_button = QPushButton("Choose Folder", self)
        self.folder_button.clicked.connect(self.choose_folder)
        layout.addWidget(self.folder_button)
        layout.addWidget(self.folder_path_label)

        # Text prompt entry
        self.text_prompt = QLineEdit(self)
        self.text_prompt.setPlaceholderText("Enter your prompt here")
        layout.addWidget(self.text_prompt)

    def choose_folder(self):
        folder_selected = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_selected:
            self.folder_path_label.setText(f"Selected Folder: {folder_selected}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
