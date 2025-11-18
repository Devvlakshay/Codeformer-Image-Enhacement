# main2.py ↔ server.py Integration

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    API Request                              │
│         POST /enhance (with image_url and user_id)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │         main2.py (Port 8000)     │
        │  ┌──────────────────────────────┐│
        │  │ 1. Download image            ││
        │  │ 2. Run CodeFormer processing ││
        │  │ 3. Create enhanced image     ││
        │  └──────────────────────────────┘│
        │           │                       │
        │           ▼                       │
        │  ┌──────────────────────────────┐│
        │  │  Call upload_to_r2()         ││
        │  │  (now uses server.py!)       ││
        │  └──────────────────────────────┘│
        └────────────┬─────────────────────┘
                     │
        ┌────────────▼─────────────────────┐
        │  HTTP POST to server.py:         │
        │  /upload-local-file-to-r2        │
        │  Data:                           │
        │  - file_path: /path/to/file.jpg  │
        │  - r2_key: uploads/123/file.jpg  │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │       server.py (Port 8001)      │
        │  ┌──────────────────────────────┐│
        │  │ 1. Validate file exists      ││
        │  │ 2. Determine content-type    ││
        │  │ 3. Upload to R2              ││
        │  │ 4. Return public URL         ││
        │  └──────────────────────────────┘│
        └────────────┬─────────────────────┘
                     │
        ┌────────────▼─────────────────────┐
        │  HTTP Response (JSON):           │
        │  {                               │
        │    "success": true,              │
        │    "public_url": "https://...", │
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
