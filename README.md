# CodeFormer - Dual-Server Image Enhancement API

A FastAPI-based image enhancement service powered by CodeFormer, with Cloudflare R2 cloud storage integration and automatic CDN cache purging.

## Architecture Overview

This project uses a **dual-server architecture** for separation of concerns:

- **main2.py (Port 8000)**: Image Enhancement Service
  - Handles image downloading, CodeFormer processing, and result coordination
  - Processes enhancement, colorization, and inpainting requests
  - Delegates all R2 uploads to server.py via HTTP

- **server.py (Port 8001)**: Unified Upload Service  
  - Centralized file upload handler with proven working code
  - Supports three upload modes: form-data files, remote URLs, local files
  - Implements Cloudflare cache purging after successful uploads
  - Manages R2 bucket operations (upload, delete)

```
┌─────────────────────────────────────────────────────────────┐
│                 API Request                                 │
│   POST /enhance (with image_url and user_id)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │      main2.py (Port 8000)        │
        │  ┌──────────────────────────────┐│
        │  │ 1. Download image            ││
        │  │ 2. CodeFormer processing     ││
        │  │ 3. Save enhanced image       ││
        │  └──────────────────────────────┘│
        │           │                       │
        │           ▼                       │
        │  HTTP POST /upload-local-file-to-r2
        │  - file_path: /tmp/enhanced.jpg  │
        │  - r2_key: uploads/user_id/file │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │      server.py (Port 8001)       │
        │  ┌──────────────────────────────┐│
        │  │ 1. Read local file           ││
        │  │ 2. Upload to R2              ││
        │  │ 3. Purge Cloudflare CDN      ││
        │  │ 4. Return public URL         ││
        │  └──────────────────────────────┘│
        └────────────┬─────────────────────┘
                     │
        ┌────────────▼─────────────────────┐
        │  HTTP Response (JSON):           │
        │  {                               │
        │    "success": true,              │
        │    "public_url": "https://...",  │
        │    "r2_key": "uploads/123/..."   │
        │  }                               │
        └─────────────────────────────────┘
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Cloudflare R2 bucket with API credentials
- Cloudflare account with cache purge API access
- GPU recommended (CUDA for faster processing)

### 2. Clone and Install Dependencies

```bash
git clone <repository-url>
cd CodeFormer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# === R2 Configuration ===
R2_ACCESS_KEY=your_r2_access_key
R2_SECRET_KEY=your_r2_secret_key
R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
R2_BUCKET=your_bucket_name
R2_PUBLIC_URL=https://cdn.yourdomain.com  # Or your R2 public URL

# === Cloudflare Cache Purge (Optional) ===
X_AUTH_EMAIL=your_cloudflare_email@example.com
X_AUTH_KEY=your_cloudflare_api_key
CLOUDFLARE_ZONE_ID=your_zone_id  # Optional: for specific zone purging

# === Server Configuration ===
LOG_LEVEL=INFO
```

### 4. Start the Services

**Terminal 1 - Start server.py (Upload Service on Port 8001):**
```bash
source venv/bin/activate
python server.py
# Server starts at http://localhost:8001
```

**Terminal 2 - Start main2.py (Enhancement Service on Port 8000):**
```bash
source venv/bin/activate
python main2.py
# API starts at http://localhost:8000
```

## API Endpoints

### main2.py (Port 8000)

#### 1. **GET** `/` - Health Check
Returns available endpoints.

```bash
curl http://localhost:8000/
```

#### 2. **GET** `/health` - Service Health
Returns server status and configuration info.

```bash
curl http://localhost:8000/health
```

#### 3. **POST** `/enhance` - Enhance Image
Enhance image quality using CodeFormer face restoration.

**Request:**
```bash
curl -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "user_id": "user_123",
    "codeformer_weight": 0.7
  }'
```

**Response:**
```json
{
  "success": true,
  "job_id": "job_abc123def456",
  "status": "processing",
  "message": "Enhancement job queued"
}
```

Then poll for results:
```bash
curl http://localhost:8000/result/job_abc123def456
```

**Response (Complete):**
```json
{
  "status": "completed",
  "public_url": "https://cdn.yourdomain.com/uploads/user_123/image.jpg",
  "r2_key": "uploads/user_123/image.jpg",
  "processing_time": 15.3,
  "file_size": 245632
}
```

**Parameters:**
- `image_url` (string, required): URL of image to enhance
- `user_id` (string, required): User identifier for organizing uploads
- `codeformer_weight` (float, optional): Weight for CodeFormer (0.0-1.0, default: 0.7)

#### 4. **POST** `/colorize` - Colorize Image
Convert grayscale images to color.

**Request:**
```bash
curl -X POST http://localhost:8000/colorize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/bw_image.jpg",
    "user_id": "user_456"
  }'
```

**Parameters:**
- `image_url` (string, required): URL of grayscale image
- `user_id` (string, required): User identifier

#### 5. **POST** `/inpaint` - Inpaint Image
Fill in masked regions of images.

**Request:**
```bash
curl -X POST http://localhost:8000/inpaint \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/masked_image.jpg",
    "user_id": "user_789"
  }'
```

**Parameters:**
- `image_url` (string, required): URL of image with mask
- `user_id` (string, required): User identifier

#### 6. **GET** `/job/{job_id}` - Check Job Status
Check processing status of a job.

```bash
curl http://localhost:8000/job/job_abc123def456
```

**Response:**
```json
{
  "job_id": "job_abc123def456",
  "status": "completed|processing|failed",
  "progress": 100,
  "error": null
}
```

#### 7. **GET** `/result/{job_id}` - Get Result
Retrieve the final result and CDN URL for a completed job.

```bash
curl http://localhost:8000/result/job_abc123def456
```

#### 8. **DELETE** `/result/{job_id}` - Delete Result
Remove a result and clean up temporary files.

```bash
curl -X DELETE http://localhost:8000/result/job_abc123def456
```

---

### server.py (Port 8001)

#### 1. **GET** `/` - Available Endpoints
Lists all available upload endpoints on this server.

```bash
curl http://localhost:8001/
```

#### 2. **POST** `/upload-file-to-r2` - Upload Form File
Upload a file via multipart form-data.

**Request:**
```bash
curl -X POST http://localhost:8001/upload-file-to-r2 \
  -F "file=@local_file.jpg" \
  -F "r2_key=uploads/user_123/file.jpg"
```

**Response:**
```json
{
  "success": true,
  "public_url": "https://cdn.yourdomain.com/uploads/user_123/file.jpg",
  "r2_key": "uploads/user_123/file.jpg",
  "file_size": 245632,
  "content_type": "image/jpeg"
}
```

#### 3. **POST** `/upload-url-to-r2` - Upload from URL
Download image from URL and upload to R2.

**Request:**
```bash
curl -X POST http://localhost:8001/upload-url-to-r2 \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "r2_key": "uploads/user_123/image.jpg"
  }'
```

**Response:**
```json
{
  "success": true,
  "public_url": "https://cdn.yourdomain.com/uploads/user_123/image.jpg",
  "r2_key": "uploads/user_123/image.jpg",
  "file_size": 524288,
  "content_type": "image/jpeg"
}
```

#### 4. **POST** `/upload-local-file-to-r2` - Upload Local File
Upload a local file from the filesystem (used by main2.py).

**Request:**
```bash
curl -X POST http://localhost:8001/upload-local-file-to-r2 \
  -d "file_path=/tmp/enhanced_image.jpg" \
  -d "r2_key=uploads/user_123/enhanced.jpg"
```

**Response:**
```json
{
  "success": true,
  "public_url": "https://cdn.yourdomain.com/uploads/user_123/enhanced.jpg",
  "r2_key": "uploads/user_123/enhanced.jpg",
  "file_size": 312456,
  "content_type": "image/jpeg"
}
```

**Parameters:**
- `file_path` (string, required): Absolute path to local file
- `r2_key` (string, required): Path in R2 bucket (e.g., `uploads/{user_id}/{filename}`)

#### 5. **DELETE** `/delete-from-r2` - Delete File
Remove a file from R2 bucket.

**Request:**
```bash
curl -X DELETE http://localhost:8001/delete-from-r2 \
  -H "Content-Type: application/json" \
  -d '{
    "r2_key": "uploads/user_123/file.jpg"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "File deleted successfully",
  "r2_key": "uploads/user_123/file.jpg"
}
```

---

## File Path Structure in R2

All uploaded files follow this naming convention in R2:

```
uploads/
├── user_123/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── enhanced_image.jpg
├── user_456/
│   ├── photo.jpg
│   └── colorized_photo.jpg
└── user_789/
    └── restored_image.png
```

**Format:** `uploads/{user_id}/{original_filename}`

This structure:
- ✅ Preserves original filenames
- ✅ Organizes files by user
- ✅ Makes bulk operations easier
- ✅ Simplifies CDN cache purging

---

## Cloudflare Integration

### Cache Purging

After successful file uploads, server.py automatically purges the Cloudflare CDN cache for the uploaded file URL.

**How it works:**
1. File uploaded to R2
2. Public URL generated: `https://cdn.yourdomain.com/uploads/user_123/file.jpg`
3. Cloudflare API called to purge cache for that URL
4. Next request gets fresh content from origin

**Requirements:**
- `X_AUTH_EMAIL`: Your Cloudflare account email
- `X_AUTH_KEY`: Cloudflare API key (not token)
- `CLOUDFLARE_ZONE_ID`: (Optional) Specific zone ID; auto-detected if not provided

### Manual Cache Purge

If needed, manually purge CDN cache:

```bash
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "X-Auth-Email: your-email@example.com" \
  -H "X-Auth-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      "https://cdn.yourdomain.com/uploads/user_123/image.jpg"
    ]
  }'
```

---

## Usage Examples

### Complete Enhancement Workflow

```bash
# Step 1: Start both servers (in separate terminals)
# Terminal 1:
python server.py

# Terminal 2:
python main2.py

# Step 2: Submit enhancement request
JOB_ID=$(curl -s -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "user_id": "user_123",
    "codeformer_weight": 0.7
  }' | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Step 3: Poll for completion (every 2 seconds)
while true; do
  STATUS=$(curl -s http://localhost:8000/job/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" == "completed" ]; then
    break
  fi
  
  sleep 2
done

# Step 4: Get result
RESULT=$(curl -s http://localhost:8000/result/$JOB_ID)
PUBLIC_URL=$(echo $RESULT | jq -r '.public_url')
echo "Enhanced image: $PUBLIC_URL"

# Step 5: Access via CDN
open "$PUBLIC_URL"  # Or curl to verify
```

---

## Troubleshooting

### Issue: "Connection refused" when connecting to server.py
- **Solution**: Ensure server.py is running on port 8001
  ```bash
  lsof -i :8001  # Check if port is in use
  ```

### Issue: R2 Upload Returns 403 (Forbidden)
- **Solution**: Verify R2 credentials in `.env`
  ```bash
  cat .env | grep R2_
  ```

### Issue: CDN Images Not Updating
- **Solution**: Cache was not purged (Cloudflare credentials may be missing)
  ```bash
  # Manually purge the specific file URL via Cloudflare dashboard
  # Or set X_AUTH_EMAIL and X_AUTH_KEY in .env for automatic purging
  ```

### Issue: main2.py Cannot Connect to server.py
- **Solution**: Both services must be running
  ```bash
  # Check both are listening:
  netstat -tuln | grep 8000
  netstat -tuln | grep 8001
  ```

### Issue: Image Processing Takes Too Long
- **Solution**: GPU acceleration may not be enabled
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  # If False, install CUDA drivers and torch with CUDA support
  ```

---

## Performance Notes

- **Enhancement (main2.py)**: 10-30 seconds per image (CPU) / 2-5 seconds (GPU)
- **Colorization**: 5-15 seconds per image
- **Inpainting**: 15-40 seconds per image
- **R2 Upload**: < 1 second for typical images
- **Cloudflare Cache Purge**: < 500ms

---

## File Descriptions

| File | Purpose |
|------|---------|
| `main2.py` | FastAPI enhancement service (Port 8000) |
| `server.py` | Unified upload service (Port 8001) |
| `inference_codeformer.py` | CodeFormer face restoration logic |
| `inference_colorization.py` | Image colorization logic |
| `inference_inpainting.py` | Image inpainting logic |
| `requirements.txt` | Python dependencies |
| `.env` | Environment configuration |

---

## Dependencies

Key packages in `requirements.txt`:
- **FastAPI** (0.104.1): Web framework
- **Uvicorn** (0.24.0): ASGI server
- **PyTorch**: Deep learning framework
- **boto3** (1.26+): AWS S3/R2 client
- **requests**: HTTP client for downloads
- **python-dotenv**: Environment variable loading
- **Pillow**: Image processing
- **torchvision**: Vision utilities

---

## License

[See LICENSE file](./LICENSE)
        │    "r2_key": "uploads/123/..."   │
        │  }                               │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │       main2.py receives          │
        │       URL and returns to         │
        │       client with               │
        │       enhanced_url               │
        └──────────────────────────────────┘
```

## Key Flow

1. **Client** → main2.py: `POST /enhance`
2. **main2.py**:
   - Downloads original image from R2 URL
   - Processes with CodeFormer
   - Saves enhanced image to local file
3. **main2.py** → **server.py**: `POST /upload-local-file-to-r2`
   - Sends local file path + R2 key
4. **server.py**:
   - Validates file exists
   - Uploads to R2 using `upload_fileobj()`
   - Returns public URL
5. **main2.py** → **Client**: Returns enhanced image URL

## Code

### In main2.py - The upload_to_r2() function

```python
def upload_to_r2(file_path: Path, r2_key: str) -> dict:
    """Delegates to server.py for uploading"""
    SERVER_URL = "http://localhost:8001/upload-local-file-to-r2"
    
    response = requests.post(
        SERVER_URL,
        data={
            "file_path": str(file_path),
            "r2_key": r2_key
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return {
            "success": True,
            "public_url": result.get("public_url"),
            "r2_key": result.get("r2_key")
        }
    else:
        return {"success": False, "error": response.text}
```

### In server.py - The new endpoint

```python
@app.post("/upload-local-file-to-r2")
async def upload_local_file_to_r2(
    file_path: str = Form(...),
    r2_key: str = Form(...)
):
    """Upload local file to R2"""
    
    # Validate file exists
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise HTTPException(status_code=400, detail="File not found")
    
    # Upload to R2
    with open(file_path_obj, 'rb') as f:
        s3.upload_fileobj(f, R2_BUCKET, r2_key)
    
    # Return URL
    public_url = f"{R2_PUBLIC_URL}/{r2_key}"
    return {
        "success": True,
        "public_url": public_url,
        "r2_key": r2_key
    }
```

## Benefits

✅ **Separation**: Upload logic centralized in server.py  
✅ **Reliable**: Uses proven working upload code from server.py  
✅ **Clear errors**: Can see exactly what failed  
✅ **Scalable**: Multiple main2.py can use one server.py  
✅ **Independent**: Can debug/restart servers separately  

## How to Use

### Start Both Servers

```bash
# Terminal 1 - Start server.py (uploads)
python server.py
# Runs on http://localhost:8001

# Terminal 2 - Start main2.py (processing)
python main2.py
# Runs on http://localhost:8000
```

### Test Image Enhancement

```bash
curl -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://cdn.qoneqt.xyz/uploads/35936/image.jpg",
    "user_id": "35936"
  }'
```

### What Happens

1. main2.py downloads the image
2. Runs CodeFormer enhancement
3. Calls `http://localhost:8001/upload-local-file-to-r2`
4. server.py uploads the enhanced image to R2
5. Returns URL to client

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to server.py` | Make sure server.py is running on port 8001 |
| `File not found` | Check if CodeFormer actually created the file |
| `R2 upload error` | Check .env has correct R2 credentials |
| `Connection refused` | Check ports: `lsof -i:8001` and `lsof -i:8000` |

## Logs to Check

```bash
# Check main2.py logs
tail -f main2.log

# Check server.py logs
tail -f server.log
```

Look for "UPLOADING TO R2 VIA server.py" in main2.py logs to confirm it's delegating correctly.
