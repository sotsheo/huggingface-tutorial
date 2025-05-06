from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
import whisper
import os
from difflib import SequenceMatcher
import shutil

app = FastAPI()
model = whisper.load_model("base")

STORE_DIR = "stores"

def transcribe(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def compare_text(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.post("/compare-audio/")
async def compare_audio(
    uploaded_file: UploadFile = File(...),
    reference_name: str = Form(...)
):
    # Kiểm tra file tham chiếu có tồn tại trong store không
    reference_path = os.path.join(STORE_DIR, reference_name)
    if not os.path.exists(reference_path):
        return JSONResponse(status_code=404, content={"error": "Reference file not found"})

    # Lưu file người dùng upload tạm thời
    temp_path = f"temp_{uploaded_file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    # Chuyển thành văn bản
    try:
        user_text = transcribe(temp_path)
        ref_text = transcribe(reference_path)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

    # So sánh
    similarity = compare_text(user_text, ref_text)

    # Dọn dẹp
    os.remove(temp_path)

    return {
        "reference_text": ref_text,
        "user_text": user_text,
        "similarity_percent": round(similarity * 100, 2)
    }
@app.post("/upload-reference/")
async def upload_reference(
    file: UploadFile = File(...),
    filename: str = Query(..., description="Tên file muốn lưu, ví dụ: hello.mp3")
):
    save_path = os.path.join(STORE_DIR, filename)

    if os.path.exists(save_path):
        return JSONResponse(status_code=400, content={"error": "File already exists in store."})

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "stored_as": filename
    }