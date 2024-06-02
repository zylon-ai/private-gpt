import io
from typing import Union

import cv2
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from injector import inject, singleton
from pdf2image import convert_from_bytes

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

@singleton
class GetOCRText:
    @inject
    def __init__(self) -> None:
        self._image = None
        self.doctr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(device)

    def _preprocess_image(self, img):
        resized_image = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def extract_text(self, cell_image: Union[None, bytes] = None, image_file: bool = False, file_path: Union[None, str] = None):
        text = ""

        if image_file:
            if file_path is None:
                raise ValueError("file_path must be provided when image_file is True.")
            pdf_file = DocumentFile.from_images(file_path)
            result = self.doctr(pdf_file)
            output = result.export()
        else:
            if cell_image is None:
                raise ValueError("cell_image must be provided when image_file is False.")

            if isinstance(cell_image, bytes):
                images = convert_from_bytes(cell_image)
                pdf_file = DocumentFile.from_images(images)
                result = self.doctr(pdf_file)
            else:
                self._image = cell_image
                preprocessed_image = self._preprocess_image(self._image)
                result = self.doctr([preprocessed_image])
                output = result.export()

        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    text += (f"{obj3['value']} ").replace("\n", "")
                text += "\n"
            text += "\n"
        if text:
            return text.strip()
        return " "
