# from paddleocr import PaddleOCR
import cv2
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from injector import singleton, inject
device = "cuda" if torch.cuda.is_available() else "cpu"

@singleton
class GetOCRText:
    @inject
    def __init__(self) -> None:
        self._image = None
        # self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.doctr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(device)

    def _preprocess_image(self, img):
        resized_image = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    ## paddleOCR
    # def extract_text(self, cell_image):
    #     text = ""
    #     self._image = cell_image
    #     preprocessd_image = self._preprocess_image(self._image)
    #     results = self.ocr.ocr(preprocessd_image, cls=True)
    #     print(results)
    #     if len(results) > 0:
    #         for result in results[0]:
    #             text += f"{result[-1][0]} "
    #     else:
    #         text = ""
    #     return text
    
    ## docTR OCR
    def extract_text(self, cell_image=None, image_file=False, file_path=None):
        text = ""
        if image_file:
            pdf_file = DocumentFile.from_images(file_path)
            result = self.doctr(pdf_file)
            output = result.export()
        else:
            self._image = cell_image
            preprocessd_image = self._preprocess_image(self._image)
            result = self.doctr([self._image])
            output = result.export()
        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    text += (f"{obj3['value']} ").replace("\n", "")
            
            text = text + "\n"
        if text:
            return text
        return " "