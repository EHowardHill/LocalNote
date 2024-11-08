from os import environ, path
import sys
import whisper
from groq import Groq
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QMessageBox, QProgressDialog, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LocalNote")

        # Initialize variables
        self.groq_key = self.load_groq_key()
        self.filepath = ""
        self.t_path = ""
        self.s_path = ""
        self.model = None  # We will load the model after user selection

        # Set up the UI
        self.initUI()

    def load_groq_key(self):
        try:
            with open("key.txt", "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""

    def save_groq_key(self, key):
        with open("key.txt", "w") as file:
            file.write(key)

    def initUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout()

        # Groq API key input
        self.groq_label = QLabel("Enter your Groq API Key:")
        self.groq_input = QLineEdit()
        self.groq_input.setText(self.groq_key)
        layout.addWidget(self.groq_label)
        layout.addWidget(self.groq_input)

        # Whisper Model Selection
        self.model_label = QLabel("Select Whisper Model:")
        layout.addWidget(self.model_label)

        # Radio Buttons for model selection
        self.model_group = QButtonGroup(self)

        self.tiny_radio = QRadioButton("tiny")
        self.base_radio = QRadioButton("base")
        self.turbo_radio = QRadioButton("turbo")

        # Add radio buttons to the group
        self.model_group.addButton(self.tiny_radio)
        self.model_group.addButton(self.base_radio)
        self.model_group.addButton(self.turbo_radio)

        # Set default selection
        self.tiny_radio.setChecked(True)

        # Add radio buttons to the layout
        layout.addWidget(self.tiny_radio)
        layout.addWidget(self.base_radio)
        layout.addWidget(self.turbo_radio)

        # Select Audio File button
        self.select_audio_btn = QPushButton("Select Audio File")
        self.select_audio_btn.clicked.connect(self.select_audio_file)
        layout.addWidget(self.select_audio_btn)

        # Start Process button
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_process)
        layout.addWidget(self.start_btn)

        # Set layout
        central_widget.setLayout(layout)

    def select_audio_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File")
        if filepath:
            self.filepath = filepath
        else:
            QMessageBox.warning(self, "Warning", "No file selected.")

    def select_transcript_path(self):
        t_path, _ = QFileDialog.getSaveFileName(self, "Save Transcript As")
        if t_path:
            self.t_path = t_path
        else:
            QMessageBox.warning(self, "Warning", "Transcript save path not selected.")

    def start_process(self):
        # Get the Groq API key from the input
        self.groq_key = self.groq_input.text()
        self.save_groq_key(self.groq_key)  # Save the key to key.txt

        environ["GROQ_API_KEY"] = self.groq_key

        if not self.filepath:
            QMessageBox.warning(self, "Warning", "Please select an audio file.")
            return

        # Get the selected model from radio buttons
        selected_model = "tiny" if self.tiny_radio.isChecked() else "base" if self.base_radio.isChecked() else "turbo"

        # Create a loading box when whisper is busy
        progress = QProgressDialog("Loading Whisper model...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        # Load the Whisper model
        try:
            self.model = whisper.load_model(selected_model)
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to load Whisper model: {e}")
            return

        progress.close()

        # Transcribe the audio file
        progress = QProgressDialog("Transcribing audio...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        absolute_path = path.abspath(self.filepath)
        try:
            mic_result = self.model.transcribe(absolute_path)
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to transcribe audio: {e}")
            return

        progress.close()

        # Prompt for transcript path
        self.select_transcript_path()

        # Save the transcript
        try:
            with open(self.t_path, "w", encoding="utf-8") as f:
                f.write(mic_result['text'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save transcript: {e}")
            return

        # Generate summary if API key is specified
        if self.groq_key:
            # Create a loading box for summary generation
            progress = QProgressDialog("Generating summary...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()

            try:
                client = Groq()
                chat_completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": "Write a summary of the following transcript: \n\n" + mic_result['text']
                        }
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1
                )
                ai_response = chat_completion.choices[0].message.content
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Error", f"Failed to generate summary: {e}")
                return

            progress.close()

            # Prompt for summary save path
            s_path, _ = QFileDialog.getSaveFileName(self, "Save Summary As")
            if s_path:
                self.s_path = s_path
                # Save the summary
                try:
                    with open(self.s_path, "w", encoding="utf-8") as f:
                        f.write(ai_response)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save summary: {e}")
                    return

        # Provide an alert window when the process is done
        QMessageBox.information(self, "Success", "Transcription has been completed" + (" and summary generated." if self.groq_key else "."))

    def closeEvent(self, event):
        # Handle any cleanup before closing the application
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
