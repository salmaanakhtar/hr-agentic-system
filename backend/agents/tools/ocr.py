from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging
import base64
import io
import os
from pathlib import Path

from . import OCRTool, ToolResult


class OCRProcessInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    image_path: Optional[str] = Field(default=None, description="Path to image file")
    language: str = Field(default="eng", description="OCR language code")


class OCRProcessOutput(BaseModel):
    text: str = Field(..., description="Extracted text from image")
    confidence: float = Field(..., description="OCR confidence score (0-1)")
    bounding_boxes: List[Dict[str, Any]] = Field(..., description="Text bounding boxes")


class OCRBatchInput(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images")
    language: str = Field(default="eng", description="OCR language code")


class OCRBatchOutput(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="OCR results for each image")


class EasyOCR(OCRTool[BaseModel, BaseModel]):

    def __init__(self, name: str, description: str, api_key: Optional[str] = None):
        super().__init__(name, description, api_key)
        self.reader = None
        self.gpu = False

    async def connect(self) -> None:
        try:
            import easyocr

            self.gpu = self._check_gpu_availability()
            self.reader = easyocr.Reader(['en'], gpu=self.gpu)

            self.logger.info(f"EasyOCR initialized, GPU: {self.gpu}")

        except ImportError:
            raise ImportError("easyocr not installed. Install with: pip install easyocr")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            raise

    def _check_gpu_availability(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def disconnect(self) -> None:
        if self.reader:

            self.reader = None
        self.logger.info("OCR tool disconnected")

    async def extract_text(self, image_data: bytes, image_format: str) -> str:
        try:
            if not self.reader:
                await self.connect()

            import numpy as np
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image)

            results = self.reader.readtext(img_array)

            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.1: 
                    text_parts.append(text)

            return ' '.join(text_parts).strip()

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""

    async def extract_text_from_file(self, file_path: str) -> str:
        try:
            if not self.reader:
                await self.connect()

            results = self.reader.readtext(file_path)

            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.1:
                    text_parts.append(text)

            return ' '.join(text_parts).strip()

        except Exception as e:
            self.logger.error(f"File OCR extraction failed: {e}")
            return ""

    def get_input_schema(self):
        return BaseModel 

    def get_output_schema(self):
        return BaseModel

    async def execute(self, input_data: BaseModel) -> ToolResult:

        return self.create_result(False, error="Use extract_text or extract_text_from_file methods directly")


class HROCRTool(EasyOCR):

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="hr_ocr",
            description="OCR tool for HR documents and forms",
            api_key=api_key
        )

    async def process_leave_request(self, image_data: str) -> Dict[str, Any]:
        result = await self.extract_text(image_data)


        text = result['text'].lower()

        parsed_data = {
            'extracted_text': result['text'],
            'confidence': result['confidence'],
            'detected_fields': {}
        }


        if 'leave' in text or 'vacation' in text:
            parsed_data['detected_fields']['request_type'] = 'leave'

        import re
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, result['text'])
        if dates:
            parsed_data['detected_fields']['dates'] = dates

        return parsed_data

    async def process_expense_receipt(self, image_data: str) -> Dict[str, Any]:
        result = await self.extract_text(image_data)


        text = result['text'].lower()

        parsed_data = {
            'extracted_text': result['text'],
            'confidence': result['confidence'],
            'detected_fields': {}
        }

        import re
        currency_pattern = r'\$?\d+\.?\d*'
        amounts = re.findall(currency_pattern, result['text'])
        if amounts:
            parsed_data['detected_fields']['amounts'] = amounts

        lines = result['text'].split('\n')
        for line in lines:
            if len(line.strip()) > 3 and not any(char.isdigit() for char in line):
                parsed_data['detected_fields']['possible_vendor'] = line.strip()
                break

        return parsed_data

    async def validate_document_quality(self, image_data: str) -> Dict[str, Any]:
        try:
            from PIL import Image
            import numpy as np

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            gray = image.convert('L')
            img_array = np.array(gray)

            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0


            quality_score = min(1.0, (brightness * 0.5 + contrast * 0.5))

            return {
                'quality_score': quality_score,
                'brightness': brightness,
                'contrast': contrast,
                'image_size': image.size,
                'is_color': image.mode != 'L',
                'recommendations': []
            }

        except Exception as e:
            return {
                'quality_score': 0.0,
                'error': str(e)
            }