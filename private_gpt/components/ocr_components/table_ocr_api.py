from fastapi import FastAPI, File, UploadFile, Response, APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from docx import Document
import os
import fitz

from private_gpt.components.ocr_components.TextExtraction import ImageToTable
from private_gpt.components.ocr_components.table_ocr import GetOCRText

upload_dir = rf"F:\LLM\privateGPT\private_gpt\uploads"

pdf_router = APIRouter(prefix="/pdf", tags=["auth"])

@pdf_router.post("/pdf_ocr")
async def get_pdf_ocr(file: UploadFile = File(...)):
    UPLOAD_DIR = upload_dir
    try:
        contents = await file.read()
    except Exception:
        return {"message": "There was an error uploading the file"}
    
    # Save the uploaded file to the dir
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    doc = Document()
    ocr = GetOCRText() 
    img_tab = ImageToTable()
    pdf_doc = fitz.open(file_path)
    for page_index in range(len(pdf_doc)): # iterate over pdf pages
        page = pdf_doc[page_index] # get the page
        image_list = page.get_images()

        for image_index, img in enumerate(image_list, start=1): # enumerate the image list
            xref = img[0]
            pix = fitz.Pixmap(pdf_doc, xref)

            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)("RGB", [pix.width, pix.height], pix.samples)
            image_path = "page_%s-image_%s.png" % (page_index, image_index)
            pix.save("page_%s-image_%s.png" % (page_index, image_index)) # save the image as png
            pixs = None
            extracted_text = ocr.extract_text(image_file=True, file_path=image_path)
            doc.add_paragraph(extracted_text)
            table_data = img_tab.table_to_csv(image_path)
            print(table_data)
            doc.add_paragraph(table_data)
            # remove image file

    doc.save(os.path.join(UPLOAD_DIR, "ocr_result.docx"))
    
    return FileResponse(path=os.path.join(UPLOAD_DIR, "ocr_result.docx"), filename="ocr_result.docx", media_type="application/pdf")

