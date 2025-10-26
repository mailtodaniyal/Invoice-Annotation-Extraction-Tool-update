# Invoice Annotation & Extraction Tool

A self-contained Python application for annotating invoice PDFs, training an NLP model, and extracting structured data into Excel.

## Features

- **Annotation Interface**: Load PDFs, draw bounding boxes, and label fields (Invoice number, Date, Total amount, Dates, Company, Meter number)
- **Model Training**: Fine-tune a DistilBERT NER model on annotations
- **Data Extraction**: Process new invoices and extract fields automatically
- **Excel Export**: Output structured data to .xlsx format
- **Offline Operation**: No external API calls, runs entirely locally

## Installation

1. Clone or download this repository
2. Install Python 3.8+ if not already installed
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python invoice_tool.py
   ```

## Usage

### 1. Annotation Phase

1. Open the application
2. Go to the "Annotate" tab
3. Click "Load PDF" to select an invoice PDF
4. Navigate pages using Previous/Next buttons
5. Select a field from the dropdown (e.g., "Invoice number")
6. Click "Annotate" to enter annotation mode
7. Draw rectangles around the text corresponding to the selected field
8. Repeat for all fields on all pages
9. Click "Save Annotations" to save your work as JSON

### 2. Training Phase

1. Go to the "Train" tab
2. Click "Train Model" to start fine-tuning the NER model
3. Wait for training to complete (progress shown in the log)

### 3. Extraction Phase

1. Go to the "Extract" tab
2. Click "Select PDFs to Process" and choose new invoice PDFs
3. Click "Extract Data" to run the model on selected PDFs
4. Click "Export to Excel" to save results to an .xlsx file

## Requirements

- Windows 10+
- Python 3.8+
- Sufficient RAM for model training (4GB+ recommended)
- GPU optional but recommended for faster training

## Dependencies

- PyQt5: GUI framework
- PyMuPDF: PDF rendering
- pdfplumber: PDF text extraction
- transformers: Hugging Face NLP models
- torch: PyTorch deep learning framework
- datasets: Data handling for training
- openpyxl: Excel file creation
- accelerate: Training optimization

## Notes

- The model uses DistilBERT for efficiency, but training requires annotated data
- Bounding box annotations are simplified; in production, consider more precise text mapping
- For best results, annotate multiple diverse invoice samples
- The extraction accuracy depends on the quality and quantity of training data

## Troubleshooting

- If PyQt5 installation fails, try: `pip install pyqt5-tools`
- For CUDA support, install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Ensure all PDFs are text-based (not image-only)