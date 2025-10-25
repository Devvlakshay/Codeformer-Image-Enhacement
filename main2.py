from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv
import uvicorn
import os
import subprocess
import shutil
from pathlib import Path
import uuid
import requests
from typing import Optional, Literal
import json
from datetime import datetime
from urllib.parse import urlparse
import mimetypes
import boto3
from io import BytesIO

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CodeFormer Image Enhancement API with R2 Storage",
    description="API for face restoration and image enhancement using CodeFormer with Cloudflare R2 storage",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
CODEFORMER_DIR = Path('/home/lakshya/Desktop/Codeformer-Image-Enhacement')
TEMP_DIR = BASE_DIR / "temp_processing"
RESULTS_DIR = BASE_DIR / "api_results"
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# R2 Configuration
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")

# Initialize R2 client
s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto"
)

# Store job status
job_status = {}


# Request Models
class EnhanceRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the image to enhance")
    upload_to_r2: bool = Field(default=True, description="Upload result to R2 bucket")


class ColorizeRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the face image to colorize")
    upload_to_r2: bool = Field(default=True, description="Upload result to R2 bucket")


class InpaintRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the masked face image")
    upload_to_r2: bool = Field(default=True, description="Upload result to R2 bucket")


# Helper Functions
def download_image(url: str, save_path: Path) -> bool:
    """Download image from URL"""
    try:
        response = requests.get(str(url), stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False


def upload_to_r2_bucket(file_path: Path, user_id: str, job_type: str) -> dict:
    """Upload processed image to R2 bucket"""
    try:
        # Create R2 key with job type prefix
        filename = file_path.name
        r2_key = f"{job_type}/{user_id}/{filename}"
        
        # Determine content type
        content_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
        
        # Upload to R2
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(
                f,
                R2_BUCKET,
                r2_key,
                ExtraArgs={"ContentType": content_type}
            )
        
        # Generate presigned URL (valid for 7 days)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": r2_key},
            ExpiresIn=604800  # 7 days
        )
        
        return {
            "success": True,
            "r2_key": r2_key,
            "presigned_url": presigned_url
        }
    except Exception as e:
        print(f"Error uploading to R2: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def run_codeformer_inference(
    input_path: Path,
    output_dir: Path,
    fidelity_weight: float = 0.7,
    has_aligned: bool = False,
    bg_upsampler: str = "realesrgan",
    face_upsample: bool = True,
    upscale: int = 2,
    detection_model: str = "retinaface_resnet50"
) -> dict:
    """Run CodeFormer inference script"""
    
    cmd = [
        "python",
        str(CODEFORMER_DIR / "inference_codeformer.py"),
        "-w", str(fidelity_weight),
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--upscale", str(upscale),
        "--detection_model", detection_model
    ]
    
    if has_aligned:
        cmd.append("--has_aligned")
    
    if bg_upsampler and bg_upsampler != "none":
        cmd.extend(["--bg_upsampler", bg_upsampler])
    
    if face_upsample:
        cmd.append("--face_upsample")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CODEFORMER_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Processing timeout (5 minutes exceeded)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_colorization(input_path: Path, output_dir: Path, upscale: int = 2) -> dict:
    """Run CodeFormer colorization script"""
    
    cmd = [
        "python",
        str(CODEFORMER_DIR / "inference_colorization.py"),
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--upscale", str(upscale)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CODEFORMER_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_inpainting(
    input_path: Path,
    output_dir: Path,
    fidelity_weight: float = 0.5,
    upscale: int = 2
) -> dict:
    """Run CodeFormer inpainting script"""
    
    cmd = [
        "python",
        str(CODEFORMER_DIR / "inference_inpainting.py"),
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "-w", str(fidelity_weight),
        "--upscale", str(upscale)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CODEFORMER_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def find_output_file(output_dir: Path, job_id: str) -> Optional[Path]:
    """Find the generated output file"""
    possible_paths = [
        output_dir / "final_results",
        output_dir / "restored_imgs",
        output_dir / "codeformer",
        output_dir
    ]
    
    for path in possible_paths:
        if path.exists():
            files = list(path.glob("*"))
            if files:
                for f in files:
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        return f
    
    return None


def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up {temp_dir}: {str(e)}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CodeFormer Image Enhancement API with R2 Storage",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "/enhance": "POST - Enhance image with face restoration (auto-uploads to R2)",
            "/colorize": "POST - Colorize black and white face images (auto-uploads to R2)",
            "/inpaint": "POST - Inpaint masked face images (auto-uploads to R2)",
            "/upload-url-to-r2": "POST - Upload image from URL directly to R2",
            "/delete-from-r2": "DELETE - Delete image from R2 bucket",
            "/job/{job_id}": "GET - Check job status",
            "/result/{job_id}": "GET - Download result file",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    codeformer_exists = CODEFORMER_DIR.exists()
    inference_script_exists = (CODEFORMER_DIR / "inference_codeformer.py").exists()
    
    # Check R2 connection
    r2_connected = False
    try:
        s3_client.head_bucket(Bucket=R2_BUCKET)
        r2_connected = True
    except Exception as e:
        print(f"R2 connection error: {str(e)}")
    
    return {
        "status": "healthy" if (codeformer_exists and inference_script_exists and r2_connected) else "unhealthy",
        "codeformer_directory": str(CODEFORMER_DIR),
        "codeformer_exists": codeformer_exists,
        "inference_script_exists": inference_script_exists,
        "r2_connected": r2_connected,
        "r2_bucket": R2_BUCKET
    }


@app.post("/enhance")
async def enhance_image(request: EnhanceRequest, background_tasks: BackgroundTasks):
    """
    Enhance image with face restoration using CodeFormer
    Automatically uploads result to R2 bucket
    """
    
    job_id = request.user_id
    
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "enhancement"
    }
    
    try:
        # Download image
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        image_filename = original_name
        input_path = user_temp_dir / image_filename
        
        job_status[job_id]["status"] = "downloading"
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run CodeFormer inference
        job_status[job_id]["status"] = "processing"
        result = run_codeformer_inference(
            input_path=input_path,
            output_dir=user_result_dir,
            fidelity_weight=0.7,
            has_aligned=False,
            bg_upsampler="realesrgan",
            face_upsample=True,
            upscale=2,
            detection_model="retinaface_resnet50"
        )
        
        if not result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = result.get("error", result.get("stderr", "Unknown error"))
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )
        
        # Find output file
        output_file = find_output_file(user_result_dir, job_id)
        
        if output_file:
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["output_file"] = str(final_output)
            
            # Upload to R2 if requested
            if request.upload_to_r2:
                job_status[job_id]["status"] = "uploading_to_r2"
                r2_result = upload_to_r2_bucket(final_output, request.user_id, "enhanced")
                
                if r2_result["success"]:
                    job_status[job_id]["r2_key"] = r2_result["r2_key"]
                    job_status[job_id]["r2_url"] = r2_result["presigned_url"]
                    job_status[job_id]["status"] = "completed"
                else:
                    job_status[job_id]["status"] = "completed_local_only"
                    job_status[job_id]["r2_error"] = r2_result.get("error")
            else:
                job_status[job_id]["status"] = "completed"
            
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        response_data = {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Image enhancement completed successfully",
            "result_url": f"/result/{job_id}"
        }
        
        if "r2_url" in job_status[job_id]:
            response_data["r2_url"] = job_status[job_id]["r2_url"]
            response_data["r2_key"] = job_status[job_id]["r2_key"]
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/colorize")
async def colorize_image(request: ColorizeRequest, background_tasks: BackgroundTasks):
    """Colorize black and white or faded face images"""
    
    job_id = request.user_id
    
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "colorization"
    }
    
    try:
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        input_path = user_temp_dir / original_name
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        result = run_colorization(input_path=input_path, output_dir=user_result_dir, upscale=2)
        
        if not result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = result.get("error", result.get("stderr", "Unknown error"))
            raise HTTPException(status_code=500, detail=f"Colorization failed: {result.get('error', 'Unknown error')}")
        
        output_file = find_output_file(user_result_dir, job_id)
        
        if output_file:
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["output_file"] = str(final_output)
            
            if request.upload_to_r2:
                job_status[job_id]["status"] = "uploading_to_r2"
                r2_result = upload_to_r2_bucket(final_output, request.user_id, "colorized")
                
                if r2_result["success"]:
                    job_status[job_id]["r2_key"] = r2_result["r2_key"]
                    job_status[job_id]["r2_url"] = r2_result["presigned_url"]
                    job_status[job_id]["status"] = "completed"
                else:
                    job_status[job_id]["status"] = "completed_local_only"
                    job_status[job_id]["r2_error"] = r2_result.get("error")
            else:
                job_status[job_id]["status"] = "completed"
            
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        response_data = {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Colorization completed successfully",
            "result_url": f"/result/{job_id}"
        }
        
        if "r2_url" in job_status[job_id]:
            response_data["r2_url"] = job_status[job_id]["r2_url"]
            response_data["r2_key"] = job_status[job_id]["r2_key"]
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/inpaint")
async def inpaint_image(request: InpaintRequest, background_tasks: BackgroundTasks):
    """Inpaint masked face images"""
    
    job_id = request.user_id
    
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "inpainting"
    }
    
    try:
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        input_path = user_temp_dir / original_name
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        result = run_inpainting(input_path=input_path, output_dir=user_result_dir, fidelity_weight=0.5, upscale=2)
        
        if not result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = result.get("error", result.get("stderr", "Unknown error"))
            raise HTTPException(status_code=500, detail=f"Inpainting failed: {result.get('error', 'Unknown error')}")
        
        output_file = find_output_file(user_result_dir, job_id)
        
        if output_file:
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["output_file"] = str(final_output)
            
            if request.upload_to_r2:
                job_status[job_id]["status"] = "uploading_to_r2"
                r2_result = upload_to_r2_bucket(final_output, request.user_id, "inpainted")
                
                if r2_result["success"]:
                    job_status[job_id]["r2_key"] = r2_result["r2_key"]
                    job_status[job_id]["r2_url"] = r2_result["presigned_url"]
                    job_status[job_id]["status"] = "completed"
                else:
                    job_status[job_id]["status"] = "completed_local_only"
                    job_status[job_id]["r2_error"] = r2_result.get("error")
            else:
                job_status[job_id]["status"] = "completed"
            
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        response_data = {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Inpainting completed successfully",
            "result_url": f"/result/{job_id}"
        }
        
        if "r2_url" in job_status[job_id]:
            response_data["r2_url"] = job_status[job_id]["r2_url"]
            response_data["r2_key"] = job_status[job_id]["r2_key"]
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ========= R2 Direct Upload/Delete Endpoints =========

@app.post("/upload-url-to-r2")
def upload_image_from_url(image_url: str = Form(...), user_id: str = Form(...)):
    """Upload image directly from a remote URL to Cloudflare R2"""
    
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {response.status_code}")
        
        parsed = urlparse(image_url)
        filename = os.path.basename(parsed.path)
        if not filename:
            raise HTTPException(status_code=400, detail="Unable to extract filename from URL")
        
        r2_key = f"uploads/{user_id}/{filename}"
        
        s3_client.upload_fileobj(
            BytesIO(response.content),
            R2_BUCKET,
            r2_key,
            ExtraArgs={"ContentType": response.headers.get("Content-Type", "image/jpeg")}
        )
        
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": r2_key},
            ExpiresIn=3600
        )
        
        return JSONResponse({
            "status": "success",
            "message": f"Image uploaded successfully to R2 at {r2_key}",
            "r2_key": r2_key,
            "presigned_url": presigned_url
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-from-r2")
def delete_from_r2(user_id: str = Form(...), filename: str = Form(...)):
    """Delete an image from R2 bucket"""
    
    try:
        r2_key = f"uploads/{user_id}/{filename}"
        
        try:
            s3_client.head_object(Bucket=R2_BUCKET, Key=r2_key)
        except:
            raise HTTPException(status_code=404, detail="File not found in R2")
        
        s3_client.delete_object(Bucket=R2_BUCKET, Key=r2_key)
        
        return JSONResponse({
            "status": "success",
            "message": f"File deleted successfully from R2: {r2_key}",
            "r2_key": r2_key
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download the result file"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status[job_id]["status"] not in ["completed", "completed_local_only"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job_status[job_id]['status']}'. Result not available."
        )
    
    output_file = Path(job_status[job_id]["output_file"])
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    media_type = mimetypes.guess_type(str(output_file))[0] or 'application/octet-stream'
    return FileResponse(output_file, media_type=media_type, filename=output_file.name)


@app.delete("/result/{job_id}")
async def delete_result(job_id: str):
    """Delete result files for a job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    user_id = job_status[job_id]["user_id"]
    result_dir = RESULTS_DIR / user_id
    
    if result_dir.exists():
        shutil.rmtree(result_dir)
    
    del job_status[job_id]
    
    return {"message": "Result deleted successfully"}


if __name__ == "__main__":
    print("=" * 80)
    print("CodeFormer FastAPI Server with R2 Storage Integration")
    print("=" * 80)
    print(f"CodeFormer Directory: {CODEFORMER_DIR}")
    print(f"Temp Directory: {TEMP_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"R2 Bucket: {R2_BUCKET}")
    print(f"R2 Endpoint: {R2_ENDPOINT}")
    print("=" * 80)
    print("\n🚀 Server starting on http://0.0.0.0:8105")
    print("📝 API Documentation: http://localhost:8105/docs")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8105, log_level="info")