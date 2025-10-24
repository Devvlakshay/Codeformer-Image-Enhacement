from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import boto3, os, uuid
from io import BytesIO

# Load environment variables
load_dotenv()

app = FastAPI(title="R2 Image Upload API")

# ======== Load R2 Configuration from .env ========
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")

# Initialize the R2 Client
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto"
)


# ========= API: Upload Image to R2 =========
@app.post("/upload-to-r2")
async def upload_to_r2(
    file: UploadFile,
    user_id: str = Form(...)
):
    """
    API to upload an image directly to Cloudflare R2.
    - Receives an image + user_id
    - Stores image in the bucket under user_id folder
    - Returns uploaded file key and (optional) public URL
    """

    try:
        # Read image into memory (no local saving)
        image_bytes = await file.read()

        # Generate unique filename
        ext = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{ext}"
        r2_key = f"uploads/{user_id}/{unique_filename}"

        # Upload directly to R2
        s3.upload_fileobj(
            BytesIO(image_bytes),
            R2_BUCKET,
            r2_key,
            ExtraArgs={"ContentType": file.content_type or "image/jpeg"}
        )

        # Optional: Generate pre-signed URL (1-hour access)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": r2_key},
            ExpiresIn=3600
        )

        return JSONResponse({
            "status": "success",
            "message": "Image uploaded successfully to R2",
            "r2_key": r2_key,
            "presigned_url": url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
