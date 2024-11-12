import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLineEdit, QListWidget,
                           QLabel, QFrame, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPalette, QColor
from text_analysis import TextAnalyzer
from ml_model import ProcessClassifier

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Gestão de Processos Jurídicos")
        self.setGeometry(100, 100, 1200, 800)
        self.text_analyzer = TextAnalyzer()
        self.process_classifier = ProcessClassifier()
        self.setup_window()
        
    def setup_window(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setMaximumWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        
        modules = ["Processos", "Relatórios", "Configurações"]
        for module in modules:
            btn = QPushButton(module)
            sidebar_layout.addWidget(btn)
        
        # Main content area
        content = QFrame()
        content.setFrameShape(QFrame.StyledPanel)
        content_layout = QVBoxLayout(content)
        
        # Search bar
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Pesquisar processos...")
        search_button = QPushButton("Buscar")
        search_layout.addWidget(self.search_bar)
        search_layout.addWidget(search_button)
        
        # Action buttons
        button_layout = QHBoxLayout()
        buttons = [
            ("Atualizar BD", self.update_database),
            ("Exportar", self.export_data),
            ("Copiar", self.copy_selected)
        ]
        for text, func in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            button_layout.addWidget(btn)
            
        # Results list
        self.results_list = QListWidget()
        
        # Add layouts to content
        content_layout.addLayout(search_layout)
        content_layout.addLayout(button_layout)
        content_layout.addWidget(self.results_list)
        
        # Chat assistant
        chat_frame = QFrame()
        chat_frame.setFrameShape(QFrame.StyledPanel)
        chat_frame.setMaximumWidth(300)
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.addWidget(QLabel("Assistente"))
        
        # Add all components to main layout
        layout.addWidget(sidebar)
        layout.addWidget(content)
        layout.addWidget(chat_frame)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QFrame {
                border-radius: 5px;
                background-color: white;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)

    def update_database(self):
        pass
        
    def export_data(self):
        pass
        
    def copy_selected(self):
        pass
        
    def display_results(self, results):
        self.results_list.clear()
        for result in results:
            self.results_list.addItem(result)
            
    def open_chat_assistant(self):
        pass

    def analyze_process(self, text):
        entities = self.text_analyzer.extract_entities(text)
        complexity = self.text_analyzer.classify_complexity(text)
        
        result = {
            'entities': entities,
            'complexity': complexity
        }
        
        self.display_analysis_results(result)
    
    def display_analysis_results(self, results):
        # Update results display in GUI
        complexity_label = QLabel(f"Complexidade: {results['complexity']}")
        parties_label = QLabel(f"Partes: {', '.join(results['entities']['parties'])}")
        dates_label = QLabel(f"Datas: {', '.join(results['entities']['dates'])}")
        
        # Add labels to results area
        self.results_layout.addWidget(complexity_label)
        self.results_layout.addWidget(parties_label)
        self.results_layout.addWidget(dates_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())