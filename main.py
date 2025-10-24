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
        "message": "CodeFormer Image Enhancement API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/enhance": "POST - Enhance image with face restoration",
            "/colorize": "POST - Colorize black and white face images",
            "/inpaint": "POST - Inpaint masked face images",
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
    
    return {
        "status": "healthy" if codeformer_exists and inference_script_exists else "unhealthy",
        "codeformer_directory": str(CODEFORMER_DIR),
        "codeformer_exists": codeformer_exists,
        "inference_script_exists": inference_script_exists
    }


@app.post("/enhance")
async def enhance_image(request: EnhanceRequest, background_tasks: BackgroundTasks):
    """
    Enhance image with face restoration using CodeFormer
    
    This endpoint:
    1. Downloads the image from the provided URL
    2. Runs CodeFormer inference with default settings
    3. Returns the job ID (user_id) for status tracking
    """
    
    # Use user_id as job_id
    job_id = request.user_id
    
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
        "type": "enhancement"
    }
    
    try:
        # Download image and preserve original filename from URL
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        image_filename = original_name
        input_path = user_temp_dir / image_filename
        
        job_status[job_id]["status"] = "downloading"
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run CodeFormer inference with default settings
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
            # Move to final location and preserve original filename
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["output_file"] = str(final_output)
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        # Schedule cleanup of temp files
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Image enhancement completed successfully" if job_status[job_id]["status"] == "completed" else "Processing failed",
            "result_url": f"/result/{job_id}" if job_status[job_id]["status"] == "completed" else None
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
    Colorize black and white or faded face images
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
        "type": "colorization"
    }
    
    try:
        # Download image and preserve original filename from URL
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        image_filename = original_name
        input_path = user_temp_dir / image_filename
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run colorization with default settings
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
        
        if output_file:
            # Move to final location and preserve original filename
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["output_file"] = str(final_output)
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Colorization completed successfully" if job_status[job_id]["status"] == "completed" else "Processing failed",
            "result_url": f"/result/{job_id}" if job_status[job_id]["status"] == "completed" else None
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
    Inpaint masked face images
    Image should have white brush marks indicating areas to inpaint
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
        "type": "inpainting"
    }
    
    try:
    # Download image and preserve original filename from URL
        original_name = os.path.basename(urlparse(str(request.image_url)).path) or f"input{Path(str(request.image_url)).suffix or '.jpg'}"
        image_filename = original_name
        input_path = user_temp_dir / image_filename
        
        if not download_image(str(request.image_url), input_path):
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Failed to download image"
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        
        # Run inpainting with default settings
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
        
        if output_file:
            # Move to final location and preserve original filename
            final_output = user_result_dir / original_name
            if final_output.exists():
                final_output.unlink()
            shutil.move(str(output_file), str(final_output))
            
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["output_file"] = str(final_output)
            job_status[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = "Output file not found"
        
        background_tasks.add_task(cleanup_temp_files, user_temp_dir)
        
        return {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": "Inpainting completed successfully" if job_status[job_id]["status"] == "completed" else "Processing failed",
            "result_url": f"/result/{job_id}" if job_status[job_id]["status"] == "completed" else None
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
    print("CodeFormer FastAPI Server")
    print("=" * 60)
    print(f"CodeFormer Directory: {CODEFORMER_DIR}")
    print(f"Temp Directory: {TEMP_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8105,
        log_level="info"
    )