import os
import fitz
import requests
from docx import Document

from fastapi import HTTPException, status, File, UploadFile, APIRouter, Request, Security, Depends
from sqlalchemy.orm import Session

from private_gpt.users import models
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.components.ocr_components.TextExtraction import ImageToTable
from private_gpt.components.ocr_components.table_ocr import GetOCRText
from private_gpt.server.ingest.ingest_router import common_ingest_logic, IngestResponse
from private_gpt.constants import OCR_UPLOAD


pdf_router = APIRouter(prefix="/pdf", tags=["ocr"])


@pdf_router.post("/pdf_ocr")
async def get_pdf_ocr(
    request: Request,
    db: Session = Depends(deps.get_db),
    file: UploadFile = File(...),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    )
):
    UPLOAD_DIR = OCR_UPLOAD
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error reading the file: {e}"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    doc = Document()
    ocr = GetOCRText()
    img_tab = ImageToTable()
    pdf_doc = fitz.open(file_path)
    # try:
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images()

        if not image_list:
            continue

        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(pdf_doc, xref)

            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)(
                    "RGB", [pix.width, pix.height], pix.samples)

            image_path = f"page_{page_index}-image_{image_index}.png"
            pix.save(image_path)
            extracted_text = ocr.extract_text(
                image_file=True, file_path=image_path)
            doc.add_paragraph(extracted_text)
            table_data = img_tab.table_to_csv(image_path)
            doc.add_paragraph(table_data)
            os.remove(image_path) 

    save_path = os.path.join(
        UPLOAD_DIR, f"{file.filename.replace('.pdf', '_ocr.docx')}")
    doc.save(save_path)

    with open(save_path, 'rb') as f:
        file_content = f.read()
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Empty file content after processing OCR"
            )
    ingested_documents = await common_ingest_logic(
        request=request,db=db, ocr_file=save_path, current_user=current_user
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)
