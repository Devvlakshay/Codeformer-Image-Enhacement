from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import boto3
import os
import requests
from urllib.parse import urlparse
from io import BytesIO
import uuid
from pathlib import Path

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Cloudflare R2 API - Upload & Delete")

# ====== R2 Configuration ======
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

# Initialize the R2 client
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto"
)


# ========= 🟢 Upload file directly =========
@app.post("/upload-file-to-r2")
async def upload_file_to_r2(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a file directly from form-data to Cloudflare R2.
    - Keeps original filename
    - Saves as uploads/{user_id}/{filename}
    """

    try:
        # --- 1. Read file content ---
        file_content = await file.read()
        
        # --- 2. Get original filename ---
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # --- 3. R2 storage key ---
        r2_key = f"uploads/{user_id}/{filename}"

        # --- 4. Determine content type ---
        content_type = file.content_type or "application/octet-stream"

        # --- 5. Upload to R2 ---
        s3.upload_fileobj(
            BytesIO(file_content),
            R2_BUCKET,
            r2_key,
            ExtraArgs={"ContentType": content_type}
        )

        # --- 6. Generate public URL ---
        public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{r2_key}" if R2_PUBLIC_URL else None

        # --- 7. Generate presigned URL (optional, for private access) ---
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": r2_key},
            ExpiresIn=3600
        )

        return JSONResponse({
            "status": "success",
            "message": f"File uploaded successfully to R2 at {r2_key}",
            "r2_key": r2_key,
            "filename": filename,
            "public_url": public_url,
            "presigned_url": presigned_url,
            "content_type": content_type,
            "size_bytes": len(file_content)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========= 🟢 Upload image from URL =========
@app.post("/upload-url-to-r2")
def upload_image_from_url(
    image_url: str = Form(...),
    user_id: str = Form(...)
):
    """
    Upload image directly from a remote URL to Cloudflare R2.
    - Keeps original filename
    - Saves as uploads/{user_id}/{filename}
    """

    try:
        # --- 1. Fetch image ---
        response = requests.get(image_url, stream=True, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {response.status_code}")

        # --- 2. Extract original filename ---
        parsed = urlparse(image_url)
        filename = os.path.basename(parsed.path)
        if not filename:
            raise HTTPException(status_code=400, detail="Unable to extract filename from URL")

        # --- 3. R2 storage key ---
        r2_key = f"uploads/{user_id}/{filename}"

        # --- 4. Upload directly to R2 ---
        s3.upload_fileobj(
            BytesIO(response.content),
            R2_BUCKET,
            r2_key,
            ExtraArgs={"ContentType": response.headers.get("Content-Type", "image/jpeg")}
        )

        # --- 5. Generate public URL ---
        public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{r2_key}" if R2_PUBLIC_URL else None

        # --- 6. Generate presigned URL (optional) ---
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": r2_key},
            ExpiresIn=3600
        )

        return JSONResponse({
            "status": "success",
            "message": f"Image uploaded successfully to R2 at {r2_key}",
            "r2_key": r2_key,
            "filename": filename,
            "public_url": public_url,
            "presigned_url": presigned_url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========= 🔴 Delete image from R2 =========
@app.delete("/delete-from-r2")
def delete_from_r2(
    user_id: str = Form(...),
    filename: str = Form(...)
):
    """
    Delete an image from R2 bucket.
    Expects: user_id and filename (original filename)
    Deletes file from uploads/{user_id}/{filename}
    """

    try:
        r2_key = f"uploads/{user_id}/{filename}"

        # --- Check if object exists ---
        try:
            s3.head_object(Bucket=R2_BUCKET, Key=r2_key)
        except s3.exceptions.ClientError:
            raise HTTPException(status_code=404, detail="File not found in R2")

        # --- Delete object ---
        s3.delete_object(Bucket=R2_BUCKET, Key=r2_key)

        return JSONResponse({
            "status": "success",
            "message": f"File deleted successfully from R2: {r2_key}",
            "r2_key": r2_key
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========= 📋 Health check =========
@app.get("/")
def root():
    return {
        "status": "running",
        "endpoints": {
            "upload_file": "/upload-file-to-r2 (POST)",
            "upload_url": "/upload-url-to-r2 (POST)",
            "delete": "/delete-from-r2 (DELETE)"
        }
    }
    
