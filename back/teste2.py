from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import os
from pdf2image import convert_from_path


app = FastAPI()

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model_name = 'impira/layoutlm-document-qa'
qa_pipeline = pipeline('document-question-answering', model=model_name)
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

def allowed_file(filename):
    file_extension = filename.rsplit(".", 1)[1].lower()
    print(f"File extension: {file_extension}")
    return "." in filename and file_extension in ALLOWED_EXTENSIONS

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    if not allowed_file('.'+file.filename):
        raise HTTPException(status_code=400, detail="Only PDF, PNG, JPG, and JPEG files are allowed for uploading.")
    file_content = await file.read()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as new_file:
        new_file.write(file_content)
    
    return {"filename": file.filename, "size_of_file": len(file_content), "stored_path": file_path}

@app.post("/ask/")
async def ask_question(file_data: dict):
    try:
        # print(os.path.join("uploads", file_data["file_path"]))
        processed_answer = qa_pipeline(image=str(os.path.join("uploads", file_data["file_path"])), question=file_data["question"])
        return {"question": file_data["question"], "answer": processed_answer[0]}
    except Exception as e:
        return {"error": str(e)}