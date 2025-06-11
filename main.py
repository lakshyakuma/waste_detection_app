from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = FastAPI()

# Load the model
model = YOLO("best.pt")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html from root directory
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save to a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        input_path = f"static/predict/{filename}"
        os.makedirs("static/predict", exist_ok=True)
        image.save(input_path)

        # 🔁 Clear old runs
        import shutil
        if os.path.exists("runs/detect"):
            shutil.rmtree("runs/detect")

        # Run prediction
        results = model.predict(source=input_path, save=True, save_txt=False, conf=0.05)

        # Get prediction classes
        predictions = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            predictions.append({
            "class": model.names[class_id],
            "confidence": round(confidence * 100, 2)
    })

        # YOLO saves to new dir like runs/detect/predict, predict2, etc.
        result_dir = results[0].save_dir  # Path object
        saved_image_path = os.path.join(result_dir, filename)
        predicted_output_path = f"static/predict/{filename}"

        # Move result image to static/predict/
        if os.path.exists(saved_image_path):
            os.replace(saved_image_path, predicted_output_path)

        return JSONResponse(content={
            "predictions": predictions,
            "image_url": f"/static/predict/{filename}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

