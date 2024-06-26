from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import os
from pdf2image import convert_from_path
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, 300)
    return images

model_name = 'impira/layoutlm-document-qa'
qa_pipeline = pipeline('document-question-answering', model=model_name)
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
def convert_pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    # Assuming you want to use the first page of the PDF
    return images[0]
def convert_to_image(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        # Convert PDF to image
        images = convert_from_path(file_path)
        # Assuming you want to use the first page of the PDF
        return images
    elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
        # Open PNG, JPG, or JPEG image directly
        return Image.open(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
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
async def ask_question(file_path: str, question: str):
    try:
        image = convert_to_image(file_path)
        
        # Process the question on the image
        processed_answer = qa_pipeline(image=image, question=question)
        
        return {"question": question, "answer": processed_answer[0]['answer']}
    except Exception as e:
        return {"error": str(e)}