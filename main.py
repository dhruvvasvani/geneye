import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import asyncio
import numpy as np
import cv2
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles

from generator import generate_synthetic_biometric
from face_extractor import extract_eye_region


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GenEye server starting…")
    yield
    print("GenEye server shutting down.")


app = FastAPI(title="GenEye", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/generate-from-face")
async def generate_from_face(
    face_image: UploadFile = File(...),
    anomaly: str = Form("none"),
    severity: float = Form(0.30),
):
    if severity < 0.0 or severity > 1.0:
        raise HTTPException(status_code=422, detail="severity must be between 0.0 and 1.0")

    contents = await face_image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image — unsupported format?")

    loop = asyncio.get_running_loop()

    try:
        eye_pil = await loop.run_in_executor(None, extract_eye_region, image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eye extraction failed: {e}")

    try:
        generated_iris = await loop.run_in_executor(
            None,
            lambda: generate_synthetic_biometric(
                anomaly_type=anomaly,
                severity=float(severity),
                base_image=eye_pil,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    buf = io.BytesIO()
    generated_iris.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, workers=1)
