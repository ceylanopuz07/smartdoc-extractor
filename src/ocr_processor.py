"""
OCR text extraction from document images
"""
from PIL import Image
import pytesseract
import base64
from io import BytesIO
import os

class OCRProcessor:
    """Extract text from document images using OCR"""
    
    def __init__(self):
        self.tesseract_path = self._find_tesseract()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            print(f"Tesseract found at: {self.tesseract_path}")
        else:
            print("Warning: Tesseract not found. OCR may not work.")
    
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
    
    def extract_text_from_image(self, image_bytes):
        """Extract text from image bytes"""
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_text_from_file(self, file_path):
        """Extract text from image file"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
