import sys
from PyQt6.QtCore import Qt, QRunnable, QThreadPool
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTextEdit,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QStyleFactory,
)

import privateGPT

class QueryRunnable(QRunnable):
    def __init__(self, query, callback):
        super().__init__()
        self.query = query
        self.callback = callback

    def run(self):
        privateGPT.answer_query(self.query, self.callback)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PrivateGPT')

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        button_layout = QHBoxLayout()

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter a query")
        self.query_input.setEnabled(True)
        self.query_input.setMinimumHeight(30)
        button_layout.addWidget(self.query_input)

        self.submit_button = QPushButton("Submit Query")
        self.submit_button.clicked.connect(self.process_query)
        self.submit_button.setEnabled(True)
        self.submit_button.setMinimumHeight(30)
        button_layout.addWidget(self.submit_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_query)
        self.reset_button.setMinimumHeight(30)
        button_layout.addWidget(self.reset_button)

        main_layout.addLayout(button_layout)

        self.answer_label = QLabel("Answer:")
        main_layout.addWidget(self.answer_label)

        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.answer_output.setMinimumHeight(300)
        main_layout.addWidget(self.answer_output)

        self.setLayout(main_layout)
        self.setMinimumSize(600,500)

    def update_answer_output(self, answer, docs):
        self.answer_output.insertPlainText(f"Answer: {answer}\n\n")
        self.answer_output.insertPlainText("Sources:\n")
        for document in docs:
            source_name = document.metadata["source"]
            page_content = document.page_content
            self.answer_output.insertPlainText(f"- {source_name}:\n{page_content}\n\n")
        self.answer_output.insertPlainText("\n\n")
        self.answer_output.ensureCursorVisible()


    def process_query(self):
        query = self.query_input.text()
        if query:
            self.answer_output.insertPlainText(f"Query: {query}\n")
            answer, docs = privateGPT.answer_query(query)
            self.update_answer_output(answer, docs)
        else:
            self.answer_output.setPlainText("Please enter a valid query.")


    def reset_query(self):
        self.query_input.clear()
        self.answer_output.clear()

def run_app():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
