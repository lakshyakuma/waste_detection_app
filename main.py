from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid
import shutil
import gc
import sys

app = FastAPI()

# Load the model once at startup
model = YOLO("best.pt")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and convert image immediately
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Ensure directories exist
        os.makedirs("static/predict", exist_ok=True)

        # Unique filename
        filename = f"{uuid.uuid4()}.jpg"
        input_path = f"static/predict/{filename}"
        image.save(input_path)

        # Free memory from original image
        del image
        gc.collect()

        # Remove old YOLO outputs
        if os.path.exists("runs/detect"):
            shutil.rmtree("runs/detect")

        # Predict (YOLO will auto-save to runs/detect/)
        results = model.predict(source=input_path, save=True, save_txt=False, conf=0.05)
        boxes = results[0].boxes
        names = model.names

        # Extract predictions
        predictions = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            predictions.append({
                "class": names[class_id],
                "confidence": round(confidence * 100, 2)
            })

        # Move image from YOLO output to static/
        result_dir = results[0].save_dir
        saved_path = os.path.join(result_dir, filename)
        output_path = f"static/predict/{filename}"

        if os.path.exists(saved_path):
            os.replace(saved_path, output_path)

        # Cleanup memory
        del results
        del boxes
        gc.collect()

        return JSONResponse(content={
            "predictions": predictions,
            "image_url": f"/static/predict/{filename}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    print("✅ App is starting...")

try:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(1)
