from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# Import your master pipeline
import main

app = FastAPI(
    title="National True 3D Cadastre API",
    description="REST API for Automated 3D ULPIN Generation and Volumetric Property Mapping",
    version="1.0.0"
)

# ==========================================
# THE FIX: Allow Cross-Origin Web Traffic
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "system": "True 3D Cadastre Engine",
        "ogc_compliant": True
    }

@app.post("/api/v1/process-cadastre")
async def process_cadastre(file: UploadFile = File(...)):
    """
    Uploads an aerial drone image or CAD blueprint, executes the True 3D pipeline,
    generates ULPINs, and compiles the WebGL model.
    """
    temp_file_path = f"temp_upload_{file.filename}"
    try:
        print(f"\n🌐 [API] Receiving payload: {file.filename}")
        
        # 1. Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Trigger the master OpenCASCADE & PostGIS pipeline!
        print(f"🌐 [API] Triggering Master Pipeline...")
        main.run_unified_cadastre_pipeline(temp_file_path)
            
        # 3. Respond to the client
        if os.path.exists("approved_cadastre.glb"):
            return JSONResponse(content={
                "status": "success",
                "message": "3D Pipeline execution fully complete. Assets registered in PostGIS.",
                "model_download_url": "http://localhost:8080/api/v1/download-model"
            })
        else:
            return JSONResponse(content={"status": "error", "message": "Pipeline failed to generate 3D model."}, status_code=500)
            
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    finally:
        # FIX: always remove the temp file, even when the pipeline raises
        # or exits early (e.g. overlap rejection). Previously, any exception
        # before the os.remove() call left temp_upload_* files on disk
        # indefinitely, and could also leave the DB engine connection open
        # if main.py's context manager was never entered.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/api/v1/download-model")
def download_model():
    """Returns the generated True 3D .glb file for web viewers."""
    file_path = "approved_cadastre.glb"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="model/gltf-binary", filename="national_cadastre.glb")
    return JSONResponse(content={"status": "error", "message": "Model not found. Run pipeline first."}, status_code=404)