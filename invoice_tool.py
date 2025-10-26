import sys
import os
import json
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTabWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QTextEdit, QSplitter
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QThread, QObject
import fitz  # PyMuPDF
import pdfplumber
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from datasets import Dataset
import openpyxl
from openpyxl import Workbook
import warnings

warnings.filterwarnings("ignore", "Some weights")
warnings.filterwarnings("ignore", "You should probably")
warnings.filterwarnings("ignore", "CropBox missing")

# Fields to extract
FIELDS = ["Invoice number", "Date", "Total amount", "Dates", "Company", "Meter number"]

# NER Labels
LABELS = ['O'] + [f'B-{f.replace(" ", "_")}' for f in FIELDS] + [f'I-{f.replace(" ", "_")}' for f in FIELDS]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

class AnnotationItem(QGraphicsRectItem):
    def __init__(self, rect, field):
        super().__init__(rect)
        self.field = field
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setBrush(QColor(255, 0, 0, 50))

class PDFViewer(QGraphicsView):
    annotation_added = pyqtSignal(QRectF, str)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.start_point = None
        self.current_rect = None
        self.current_field = None

    def set_pdf_page(self, pixmap):
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def set_current_field(self, field):
        self.current_field = field

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_field:
            self.start_point = self.mapToScene(event.pos())
            self.current_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            self.current_rect.setPen(QPen(QColor(255, 0, 0), 2))
            self.current_rect.setBrush(QColor(255, 0, 0, 50))
            self.scene.addItem(self.current_rect)

    def mouseMoveEvent(self, event):
        if self.current_rect and self.start_point:
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()
            self.current_rect.setRect(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_rect and self.current_field:
            rect = self.current_rect.rect()
            self.annotation_added.emit(rect, self.current_field)
            self.current_rect = None
            self.start_point = None

class TrainingWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, annotations, model_name="distilbert-base-uncased"):
        super().__init__()
        self.annotations = annotations
        self.model_name = model_name

    def run(self):
        try:
            # Prepare dataset
            texts = []
            labels = []
            for ann in self.annotations:
                with pdfplumber.open(ann['pdf_path']) as pdf:
                    page = pdf.pages[ann['page']]
                    text = page.extract_text()
                    # Simple mapping: assume bounding box corresponds to text spans
                    # In real implementation, need better text extraction mapping
                    texts.append(text)
                    label = [0] * len(text.split())  # Use integer labels
                    # For simplicity, mark first word as entity
                    field_label = f'B-{ann["field"].replace(" ", "_")}'
                    label[0] = label2id[field_label]
                    labels.append(label)

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, timeout=30)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(LABELS),
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True
            )

            # Tokenize and align labels
            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
                labels = []
                for i, label in enumerate(examples["ner_tags"]):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    labels.append(label_ids)
                tokenized_inputs["labels"] = labels
                return tokenized_inputs

            # Create dataset
            dataset = Dataset.from_dict({"tokens": [t.split() for t in texts], "ner_tags": labels})
            tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

            # Training
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                save_steps=10_000,
                save_total_limit=2,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            trainer.train()
            model.save_pretrained("./trained_model")
            tokenizer.save_pretrained("./trained_model")

            self.finished.emit("Training completed successfully!")
        except Exception as e:
            self.finished.emit(f"Training failed: {str(e)}")

class ExtractionWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, pdf_paths, model_path="./trained_model"):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.model_path = model_path

    def run(self):
        try:
            nlp = pipeline("ner", model=self.model_path, tokenizer=self.model_path)
            results = []
            for i, pdf_path in enumerate(self.pdf_paths):
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"

                entities = nlp(text)
                extracted = {field: "" for field in FIELDS}
                for entity in entities:
                    label = entity['entity'].split('-')[-1].replace('_', ' ')
                    if label in FIELDS:
                        extracted[label] = entity['word']

                results.append(extracted)
                self.progress.emit(int((i + 1) / len(self.pdf_paths) * 100))

            self.finished.emit(results)
        except Exception as e:
            self.finished.emit([])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invoice Annotation & Extraction Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.annotations = []
        self.current_pdf_path = None
        self.current_page = 0

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Annotation Tab
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)

        # PDF controls
        pdf_controls = QHBoxLayout()
        self.load_pdf_btn = QPushButton("Load PDF")
        self.load_pdf_btn.clicked.connect(self.load_pdf)
        pdf_controls.addWidget(self.load_pdf_btn)

        self.page_label = QLabel("Page: 1")
        pdf_controls.addWidget(self.page_label)

        self.prev_page_btn = QPushButton("Previous")
        self.prev_page_btn.clicked.connect(self.prev_page)
        pdf_controls.addWidget(self.prev_page_btn)

        self.next_page_btn = QPushButton("Next")
        self.next_page_btn.clicked.connect(self.next_page)
        pdf_controls.addWidget(self.next_page_btn)

        annotation_layout.addLayout(pdf_controls)

        # Viewer and controls
        viewer_layout = QHBoxLayout()
        self.pdf_viewer = PDFViewer()
        self.pdf_viewer.annotation_added.connect(self.add_annotation)
        viewer_layout.addWidget(self.pdf_viewer)

        controls_layout = QVBoxLayout()
        self.field_combo = QComboBox()
        self.field_combo.addItems(FIELDS)
        controls_layout.addWidget(QLabel("Select Field:"))
        controls_layout.addWidget(self.field_combo)

        self.annotate_btn = QPushButton("Annotate")
        self.annotate_btn.clicked.connect(self.set_annotation_mode)
        controls_layout.addWidget(self.annotate_btn)

        self.save_annotations_btn = QPushButton("Save Annotations")
        self.save_annotations_btn.clicked.connect(self.save_annotations)
        controls_layout.addWidget(self.save_annotations_btn)

        self.load_annotations_btn = QPushButton("Load Annotations")
        self.load_annotations_btn.clicked.connect(self.load_annotations)
        controls_layout.addWidget(self.load_annotations_btn)

        self.annotations_list = QListWidget()
        controls_layout.addWidget(QLabel("Annotations:"))
        controls_layout.addWidget(self.annotations_list)

        viewer_layout.addLayout(controls_layout)
        annotation_layout.addLayout(viewer_layout)

        self.tabs.addTab(annotation_tab, "Annotate")

        # Training Tab
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        training_layout.addWidget(self.train_btn)

        self.training_progress = QProgressBar()
        training_layout.addWidget(self.training_progress)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        training_layout.addWidget(self.training_log)

        self.tabs.addTab(training_tab, "Train")

        # Extraction Tab
        extraction_tab = QWidget()
        extraction_layout = QVBoxLayout(extraction_tab)

        self.select_pdfs_btn = QPushButton("Select PDFs to Process")
        self.select_pdfs_btn.clicked.connect(self.select_pdfs)
        extraction_layout.addWidget(self.select_pdfs_btn)

        self.extract_btn = QPushButton("Extract Data")
        self.extract_btn.clicked.connect(self.start_extraction)
        extraction_layout.addWidget(self.extract_btn)

        self.extraction_progress = QProgressBar()
        extraction_layout.addWidget(self.extraction_progress)

        self.export_excel_btn = QPushButton("Export to Excel")
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        extraction_layout.addWidget(self.export_excel_btn)

        self.tabs.addTab(extraction_tab, "Extract")

    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.current_pdf_path = file_path
            self.current_page = 0
            self.display_page()

    def display_page(self):
        if self.current_pdf_path:
            doc = fitz.open(self.current_pdf_path)
            page = doc.load_page(self.current_page)
            pix = page.get_pixmap()
            img_path = tempfile.mktemp(suffix='.png')
            pix.save(img_path)
            pixmap = QPixmap(img_path)
            self.pdf_viewer.set_pdf_page(pixmap)
            self.page_label.setText(f"Page: {self.current_page + 1}/{doc.page_count}")
            doc.close()
            os.remove(img_path)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_page()

    def next_page(self):
        doc = fitz.open(self.current_pdf_path)
        if self.current_page < doc.page_count - 1:
            self.current_page += 1
            self.display_page()
        doc.close()

    def set_annotation_mode(self):
        field = self.field_combo.currentText()
        self.pdf_viewer.set_current_field(field)

    def add_annotation(self, rect, field):
        # Convert to PDF coordinates (simplified)
        item = AnnotationItem(rect, field)
        self.pdf_viewer.scene.addItem(item)
        self.annotations.append({
            'pdf_path': self.current_pdf_path,
            'page': self.current_page,
            'rect': [rect.x(), rect.y(), rect.width(), rect.height()],
            'field': field
        })
        self.annotations_list.addItem(f"{field}: {rect}")

    def save_annotations(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.annotations, f)

    def load_annotations(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                self.annotations = json.load(f)
            self.update_annotations_list()

    def update_annotations_list(self):
        self.annotations_list.clear()
        for ann in self.annotations:
            self.annotations_list.addItem(f"{ann['field']}: {ann['rect']}")

    def start_training(self):
        if not self.annotations:
            QMessageBox.warning(self, "Warning", "No annotations available for training.")
            return

        self.training_worker = TrainingWorker(self.annotations)
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        self.training_worker.progress.connect(self.training_progress.setValue)
        self.training_worker.finished.connect(self.training_finished)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_thread.start()

    def training_finished(self, message):
        self.training_log.append(message)
        self.training_thread.quit()

    def select_pdfs(self):
        self.pdfs_to_process, _ = QFileDialog.getOpenFileNames(self, "Select PDFs", "", "PDF Files (*.pdf)")

    def start_extraction(self):
        if not self.pdfs_to_process:
            QMessageBox.warning(self, "Warning", "No PDFs selected for processing.")
            return

        self.extraction_worker = ExtractionWorker(self.pdfs_to_process)
        self.extraction_thread = QThread()
        self.extraction_worker.moveToThread(self.extraction_thread)
        self.extraction_worker.progress.connect(self.extraction_progress.setValue)
        self.extraction_worker.finished.connect(self.extraction_finished)
        self.extraction_thread.started.connect(self.extraction_worker.run)
        self.extraction_thread.start()

    def extraction_finished(self, results):
        self.extracted_data = results
        QMessageBox.information(self, "Info", f"Extraction completed for {len(results)} PDFs.")
        self.extraction_thread.quit()

    def export_to_excel(self):
        if not hasattr(self, 'extracted_data'):
            QMessageBox.warning(self, "Warning", "No extracted data available.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            wb = Workbook()
            ws = wb.active
            ws.append(FIELDS)
            for row in self.extracted_data:
                ws.append([row[field] for field in FIELDS])
            wb.save(file_path)
            QMessageBox.information(self, "Info", "Data exported to Excel successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())