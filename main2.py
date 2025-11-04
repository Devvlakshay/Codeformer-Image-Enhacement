from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
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
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CodeFormer Image Enhancement API",
    description="API for face restoration and image enhancement using CodeFormer inference scripts",
    version="2.0.0"
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
CODEFORMER_DIR = Path('/Users/lakshyaborasi/Desktop/CodeFormer')
TEMP_DIR = BASE_DIR / "temp_processing"
RESULTS_DIR = BASE_DIR / "api_results"
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# R2 Configuration
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

# Initialize R2 client
def get_r2_client():
    """Initialize and return R2 client"""
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )

# Store job status
job_status = {}


# Request Models
class EnhanceRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the image to enhance")


class ColorizeRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the face image to colorize")


class InpaintRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="URL of the masked face image")


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


def parse_r2_url(url: str) -> dict:
    """
    Parse R2 URL to extract directory, user_id, and filename
    Example: https://cdn.qoneqt.xyz/uploads/35936/faceverify_iShvvbGQ5Q.jpg
    Returns: {
        "directory": "uploads",
        "user_id": "35936",
        "filename": "faceverify_iShvvbGQ5Q.jpg",
        "full_key": "uploads/35936/faceverify_iShvvbGQ5Q.jpg"
    }
    """
    try:
        parsed = urlparse(url)
        # Remove leading slash and split path
        path_parts = parsed.path.lstrip('/').split('/')
        
        if len(path_parts) >= 3:
            return {
                "directory": path_parts[0],  # e.g., "uploads"
                "user_id": path_parts[1],     # e.g., "35936"
                "filename": path_parts[2],    # e.g., "faceverify_iShvvbGQ5Q.jpg"
                "full_key": '/'.join(path_parts)  # Full S3 key
            }
        else:
            raise ValueError("Invalid URL format. Expected: {base_url}/{directory}/{user_id}/{filename}")
    except Exception as e:
        print(f"Error parsing R2 URL: {str(e)}")
        return None


def upload_to_r2(file_path: Path, r2_key: str) -> dict:
    """
    Upload file to R2 bucket
    Returns: dict with success status and public URL
    """
    try:
        r2_client = get_r2_client()
        
        # Determine content type
        content_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
        
        # Upload file
        with open(file_path, 'rb') as f:
            r2_client.put_object(
                Bucket=R2_BUCKET,
                Key=r2_key,
                Body=f,
                ContentType=content_type
            )
        
        # Construct public URL
        public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{r2_key}"
        
        return {
            "success": True,
            "public_url": public_url,
            "r2_key": r2_key
        }
    except ClientError as e:
        print(f"R2 upload error: {str(e)}")
        return {
            "success": False,
            "error": f"R2 upload failed: {str(e)}"
        }
    except Exception as e:
        print(f"Unexpected error uploading to R2: {str(e)}")
        return {
            "success": False,
            "error": f"Upload failed: {str(e)}"
        }


def delete_from_r2(r2_key: str) -> bool:
    """Delete file from R2 bucket (optional, for cleanup)"""
    try:
        r2_client = get_r2_client()
        r2_client.delete_object(Bucket=R2_BUCKET, Key=r2_key)
        return True
    except Exception as e:
        print(f"Error deleting from R2: {str(e)}")
        return False


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
            timeout=300  # 5 minutes timeout
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
    # CodeFormer typically saves to results/final_results or results/restored_imgs
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
                # Return the first image file found
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
        "message": "CodeFormer Image Enhancement API with R2 Integration",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/enhance": "POST - Enhance image with face restoration and upload to R2",
            "/colorize": "POST - Colorize black and white face images and upload to R2",
            "/inpaint": "POST - Inpaint masked face images and upload to R2",
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
    r2_configured = all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET])
    r2_accessible = False
    
    if r2_configured:
        try:
            r2_client = get_r2_client()
            r2_client.head_bucket(Bucket=R2_BUCKET)
            r2_accessible = True
        except Exception as e:
            print(f"R2 health check failed: {str(e)}")
    
    return {
        "status": "healthy" if (codeformer_exists and inference_script_exists and r2_accessible) else "degraded",
        "codeformer_directory": str(CODEFORMER_DIR),
        "codeformer_exists": codeformer_exists,
        "inference_script_exists": inference_script_exists,
        "r2_configured": r2_configured,
        "r2_accessible": r2_accessible,
        "r2_bucket": R2_BUCKET if r2_configured else None
    }


@app.post("/enhance")
async def enhance_image(request: EnhanceRequest, background_tasks: BackgroundTasks):
    """
    Enhance image with face restoration using CodeFormer and upload to R2
    
    This endpoint:
    1. Downloads the image from the provided URL
    2. Parses the R2 path from the URL
    3. Runs CodeFormer inference with default settings
    4. Uploads the enhanced image back to R2 (replaces original)
    5. Returns the new R2 URL
    """
    
    # Use user_id as job_id
    job_id = request.user_id
    
    # Parse R2 URL to get the storage path
    r2_info = parse_r2_url(str(request.image_url))
    if not r2_info:
        raise HTTPException(status_code=400, detail="Invalid R2 URL format")
    
    # Create user-specific directories
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize job status
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "enhancement",
        "original_url": str(request.image_url),
        "r2_key": r2_info["full_key"]
    }
    
    try:
        # Download image
        image_filename = r2_info["filename"]
        input_path = user_temp_dir / image_filename
        
        job_status[job_id]["status"] = "downloading"
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run CodeFormer inference
        job_status[job_id]["status"] = "enhancing"
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
        
        if not output_file:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
            raise HTTPException(status_code=500, detail="Output file not found after processing")
        
        # Move to final location with original filename
        final_output = user_result_dir / image_filename
        if final_output.exists():
            final_output.unlink()
        shutil.move(str(output_file), str(final_output))
        
        # Upload to R2 (replaces existing file)
        job_status[job_id]["status"] = "uploading"
        upload_result = upload_to_r2(final_output, r2_info["full_key"])
        
        if not upload_result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = upload_result.get("error", "R2 upload failed")
            raise HTTPException(status_code=500, detail=upload_result.get("error", "R2 upload failed"))
        
        # Update job status
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["output_file"] = str(final_output)
        job_status[job_id]["r2_url"] = upload_result["public_url"]
        job_status[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Image enhancement completed and uploaded to R2 successfully",
            "original_url": str(request.image_url),
            "enhanced_url": upload_result["public_url"],
            "r2_key": r2_info["full_key"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/colorize")
async def colorize_image(request: ColorizeRequest, background_tasks: BackgroundTasks):
    """
    Colorize black and white or faded face images and upload to R2
    """
    
    job_id = request.user_id
    
    # Parse R2 URL
    r2_info = parse_r2_url(str(request.image_url))
    if not r2_info:
        raise HTTPException(status_code=400, detail="Invalid R2 URL format")
    
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "colorization",
        "original_url": str(request.image_url),
        "r2_key": r2_info["full_key"]
    }
    
    try:
        # Download image
        image_filename = r2_info["filename"]
        input_path = user_temp_dir / image_filename
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run colorization
        job_status[job_id]["status"] = "colorizing"
        result = run_colorization(
            input_path=input_path,
            output_dir=user_result_dir,
            upscale=2
        )
        
        if not result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = result.get("error", result.get("stderr", "Unknown error"))
            raise HTTPException(status_code=500, detail=f"Colorization failed: {result.get('error', 'Unknown error')}")
        
        # Find and move output file
        output_file = find_output_file(user_result_dir, job_id)
        
        if not output_file:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
            raise HTTPException(status_code=500, detail="Output file not found")
        
        final_output = user_result_dir / image_filename
        if final_output.exists():
            final_output.unlink()
        shutil.move(str(output_file), str(final_output))
        
        # Upload to R2
        job_status[job_id]["status"] = "uploading"
        upload_result = upload_to_r2(final_output, r2_info["full_key"])
        
        if not upload_result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = upload_result.get("error", "R2 upload failed")
            raise HTTPException(status_code=500, detail=upload_result.get("error", "R2 upload failed"))
        
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["output_file"] = str(final_output)
        job_status[job_id]["r2_url"] = upload_result["public_url"]
        job_status[job_id]["completed_at"] = datetime.now().isoformat()
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Colorization completed and uploaded to R2 successfully",
            "original_url": str(request.image_url),
            "enhanced_url": upload_result["public_url"],
            "r2_key": r2_info["full_key"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/inpaint")
async def inpaint_image(request: InpaintRequest, background_tasks: BackgroundTasks):
    """
    Inpaint masked face images and upload to R2
    Image should have white brush marks indicating areas to inpaint
    """
    
    job_id = request.user_id
    
    # Parse R2 URL
    r2_info = parse_r2_url(str(request.image_url))
    if not r2_info:
        raise HTTPException(status_code=400, detail="Invalid R2 URL format")
    
    user_temp_dir = TEMP_DIR / request.user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    
    user_result_dir = RESULTS_DIR / request.user_id
    user_result_dir.mkdir(parents=True, exist_ok=True)
    
    job_status[job_id] = {
        "status": "processing",
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "type": "inpainting",
        "original_url": str(request.image_url),
        "r2_key": r2_info["full_key"]
    }
    
    try:
        # Download image
        image_filename = r2_info["filename"]
        input_path = user_temp_dir / image_filename
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run inpainting
        job_status[job_id]["status"] = "inpainting"
        result = run_inpainting(
            input_path=input_path,
            output_dir=user_result_dir,
            fidelity_weight=0.5,
            upscale=2
        )
        
        if not result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = result.get("error", result.get("stderr", "Unknown error"))
            raise HTTPException(status_code=500, detail=f"Inpainting failed: {result.get('error', 'Unknown error')}")
        
        # Find and move output file
        output_file = find_output_file(user_result_dir, job_id)
        
        if not output_file:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
            raise HTTPException(status_code=500, detail="Output file not found")
        
        final_output = user_result_dir / image_filename
        if final_output.exists():
            final_output.unlink()
        shutil.move(str(output_file), str(final_output))
        
        # Upload to R2
        job_status[job_id]["status"] = "uploading"
        upload_result = upload_to_r2(final_output, r2_info["full_key"])
        
        if not upload_result["success"]:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = upload_result.get("error", "R2 upload failed")
            raise HTTPException(status_code=500, detail=upload_result.get("error", "R2 upload failed"))
        
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["output_file"] = str(final_output)
        job_status[job_id]["r2_url"] = upload_result["public_url"]
        job_status[job_id]["completed_at"] = datetime.now().isoformat()
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Inpainting completed and uploaded to R2 successfully",
            "original_url": str(request.image_url),
            "enhanced_url": upload_result["public_url"],
            "r2_key": r2_info["full_key"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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
    
    if job_status[job_id]["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job_status[job_id]['status']}'. Result not available."
        )
    
    output_file = Path(job_status[job_id]["output_file"])
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Determine media type from filename
    media_type = mimetypes.guess_type(str(output_file))[0] or 'application/octet-stream'
    return FileResponse(
        output_file,
        media_type=media_type,
        filename=output_file.name
    )


@app.delete("/result/{job_id}")
async def delete_result(job_id: str):
    """Delete result files for a job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    user_id = job_status[job_id]["user_id"]
    result_dir = RESULTS_DIR / user_id / job_id
    
    if result_dir.exists():
        shutil.rmtree(result_dir)
    
    del job_status[job_id]
    
    return {"message": "Result deleted successfully"}


if __name__ == "__main__":
    print("=" * 60)
    print("CodeFormer FastAPI Server with R2 Integration")
    print("=" * 60)
    print(f"CodeFormer Directory: {CODEFORMER_DIR}")
    print(f"Temp Directory: {TEMP_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"R2 Bucket: {R2_BUCKET}")
    print(f"R2 Public URL: {R2_PUBLIC_URL}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8105,
        log_level="info"
    )