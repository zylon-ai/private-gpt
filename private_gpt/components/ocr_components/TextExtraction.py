import io
import cv2
import csv
import torch 
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection

from typing import Literal

from TextExtraction import GetOCRText

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size
    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image


class ImageToTable:
    def __init__(self, tokens:list=None, detection_class_thresholds:dict=None) -> None:
        self._table_model = "microsoft/table-transformer-detection"
        self._structure_model = "microsoft/table-structure-recognition-v1.1-all"
        self._image = None
        self._table_image = None
        self._file_path = None
        self.text_data =[]
        self.tokens = []
        self.detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }
        # for ocr stuffs
        self.get_ocr = GetOCRText()

    def _prepare_for_nn_input(self, image):
        structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        pixel_values = structure_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(device)
        return pixel_values


    def _detection(self, detection_type: Literal['table', 'table_structure'], image):
        if detection_type == "table":
            model = AutoModelForObjectDetection.from_pretrained(self._table_model)
        elif detection_type == "table_structure":
            model = TableTransformerForObjectDetection.from_pretrained(self._structure_model)
        pixel_values = self._prepare_for_nn_input(image)
        pixel_values = pixel_values.to(device)
        model.to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"
        return outputs, id2label   
 
    
    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=10):
        tables_crop = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue
            cropped_table = {}
            bbox = obj['bbox']
            bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
            cropped_img = img.crop(bbox)
            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0]-bbox[0],
                                token['bbox'][1]-bbox[1],
                                token['bbox'][2]-bbox[0],
                                token['bbox'][3]-bbox[1]]
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0]-bbox[3]-1,
                            bbox[0],
                            cropped_img.size[0]-bbox[1]-1,
                            bbox[2]]
                    token['bbox'] = bbox
            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens
            tables_crop.append(cropped_table)
        return tables_crop

    def outputs_to_objects(self, outputs, img_size, id2label):
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)
        def rescale_bboxes(out_bbox, size):
            img_w, img_h = size
            b = box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]
        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})
        return objects

    def get_cell_coordinates_by_row(self, table_data):
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])
        def find_cell_coordinates(row, column):
            cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
            return cell_bbox
        cell_coordinates = []
        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})
            row_cells.sort(key=lambda x: x['column'][0])
            cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})
        cell_coordinates.sort(key=lambda x: x['row'][1])
        return cell_coordinates
    
    def apply_ocr(self, cell_coordinates):
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                print(cell)
                cell_image = np.array(self._table_image.crop(cell["cell"]))
                result = self.get_ocr.extract_text(np.array(cell_image))
                row_text.append(result)
            self.text_data.append(row_text)
        
    
    def table_to_csv(self, image_path):
        self._image = Image.open(image_path).convert("RGB")
        outputs, id2label = self._detection(detection_type='table', image=self._image)
        objects = self.outputs_to_objects(outputs=outputs, img_size=self._image.size, id2label=id2label)
        tables_crop = self.objects_to_crops(self._image, self.tokens, objects, self.detection_class_thresholds, padding=0)
        for table_crop in tables_crop:
            cropped_table = table_crop['image'].convert("RGB")
            self._table_image = cropped_table
            resized_image = self._prepare_for_nn_input(cropped_table)
            outputs, structure_id2label = self._detection(detection_type='table_structure', image=cropped_table)
            cells = self.outputs_to_objects(outputs, cropped_table.size, structure_id2label)
            cell_coordinates = self.get_cell_coordinates_by_row(cells)
            self.apply_ocr(cell_coordinates)
        if self.text_data:
            # print(self.text_data)
            return "".join(",".join(row) for row in self.text_data)
        return ""




        