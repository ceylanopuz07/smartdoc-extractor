"""
OCR text extraction from document images and PDFs
"""
from PIL import Image
import pytesseract
import base64
from io import BytesIO
import os
import tempfile
from pdf2image import convert_from_bytes
import PyPDF2

class OCRProcessor:
    """Extract text from document images using OCR"""
    
    def __init__(self):
        self.tesseract_path = self._find_tesseract()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            print(f"Tesseract found at: {self.tesseract_path}")
        else:
            print("Warning: Tesseract not found. OCR may not work.")
        
        # Set poppler path for PDF processing
        self.poppler_path = self._find_poppler()
        if self.poppler_path:
            os.environ['PATH'] = self.poppler_path + os.pathsep + os.environ.get('PATH', '')
            print(f"Poppler found at: {self.poppler_path}")
        else:
            print("Warning: Poppler not found. PDF processing may not work.")
    
    def _find_tesseract(self):
        """Find Tesseract executable"""
        common_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _find_poppler(self):
        """Find Poppler executable directory"""
        common_paths = [
            '/opt/homebrew/bin',
            '/usr/bin',
            '/usr/local/bin',
            '/opt/homebrew/Cellar/poppler/26.07.0/bin',
            'C:\\Program Files\\poppler-0.68.0\\bin',
            'C:\\Program Files (x86)\\poppler-0.68.0\\bin'
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None
    
    def extract_text_from_image(self, image_bytes):
        """Extract text from image bytes"""
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_bytes):
        """Extract text from PDF bytes"""
        try:
            # First try to extract text directly from PDF
            text = self._extract_text_from_pdf_bytes(pdf_bytes)
            if text and len(text.strip()) > 50:
                return text
            
            # If direct extraction fails, convert PDF to images and use OCR
            print("Direct PDF text extraction failed, using OCR...")
            return self._ocr_pdf_bytes(pdf_bytes)
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def _extract_text_from_pdf_bytes(self, pdf_bytes):
        """Extract text directly from PDF bytes"""
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Direct PDF text extraction error: {e}")
            return ""
    
    def _ocr_pdf_bytes(self, pdf_bytes):
        """Convert PDF to images and extract text using OCR"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text
        except Exception as e:
            print(f"PDF OCR error: {e}")
            return ""
    
    def extract_text_from_file(self, file_path):
        """Extract text from image or PDF file"""
        try:
            # Check if file is PDF
            if file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_bytes = f.read()
                return self.extract_text_from_pdf(pdf_bytes)
            else:
                # Handle as image
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_text(self, file_bytes, filename):
        """Extract text from file bytes based on file type"""
        if filename.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        else:
            return self.extract_text_from_image(file_bytes)
