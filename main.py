from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
import os, uuid
import cv2
import numpy as np

main = FastAPI(title="Servidor de Análisis de Mastitis")

# 🔒 CORS
main.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000", "http://192.168.18.82:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Configuración
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
main.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 🧠 Modelo
model = YOLO("models/best.pt")

# 📏 Parámetros
MIN_CONFIDENCE = 0.55
MIN_BOX_SIZE = 0.08
MAX_BOX_SIZE = 0.60
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 2.0


# 🔹 ERROR
def build_error(position, filename, error, confidence=0):
    return {
        "image_position": position,
        "filename": filename,
        "valid": False,
        "error": error,
        "image_path": None,
        "box": None,
        "confidence": confidence
    }


# 🔹 VALIDACIÓN EN MEMORIA
def validate_image(img):
    if img is None:
        return False, "No se pudo leer la imagen"

    h, w = img.shape[:2]
    if w < 320 or h < 320:
        return False, "Imagen muy pequeña"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 40 or brightness > 210:
        return False, "Problema de iluminación"

    # ⚡ OPCIONAL (puedes comentar para más velocidad)
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # if laplacian.var() < 150:
    #     return False, "Imagen borrosa"

    return True, img


# 🔹 VALIDACIÓN DE UBRE
def validate_udder_detection(box, img_w, img_h):
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

    box_w, box_h = x2 - x1, y2 - y1
    box_area = (box_w * box_h) / (img_w * img_h)

    if box_area < MIN_BOX_SIZE:
        return False, "Muy pequeña"
    if box_area > MAX_BOX_SIZE:
        return False, "Muy grande"

    aspect_ratio = box_h / box_w
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return False, "Forma no válida"

    return True, (x1, y1, x2, y2, box_w, box_h)


# 🚀 ENDPOINT
@main.post("/analyze")
async def analyze(animal_id: str = Form(...), files: list[UploadFile] = File(...)):

    if not animal_id:
        raise HTTPException(400, "animal_id es requerido")
    if not files:
        raise HTTPException(400, "Debe subir imágenes")
    if len(files) > 2:
        raise HTTPException(400, "Máximo 2 imágenes")

    invalid_images = []
    valid_images = []

    # 🔹 PREPROCESAMIENTO EN MEMORIA
    for i, file in enumerate(files, start=1):

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            invalid_images.append(build_error(i, file.filename, "Formato no soportado"))
            continue

        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        valid, result = validate_image(img)

        if not valid:
            invalid_images.append(build_error(i, file.filename, result))
        else:
            # 🔥 REDUCIR TAMAÑO (CLAVE)
            img = cv2.resize(result, (640, 640))
            valid_images.append((i, file.filename, img))

    if invalid_images:
        raise HTTPException(400, {
            "message": "Imágenes inválidas",
            "details": invalid_images
        })

    results_data = []

    # 🔹 YOLO EN MEMORIA
    for i, filename, img in valid_images:
        try:
            results = model.predict(img, imgsz=640, conf=0.3, verbose=False)

            if not results[0].boxes:
                results_data.append(build_error(i, filename, "No se detecta ubre"))
                continue

            box = results[0].boxes[int(results[0].boxes.conf.argmax())]
            confidence = float(box.conf[0])

            if confidence < MIN_CONFIDENCE:
                results_data.append(build_error(i, filename, "Baja confianza", round(confidence*100,2)))
                continue

            h, w = img.shape[:2]
            valid_box, box_data = validate_udder_detection(box, w, h)

            if not valid_box:
                results_data.append(build_error(i, filename, box_data, round(confidence*100,2)))
                continue

            # ✅ GUARDAR SOLO SI ES VÁLIDA
            cls_id = int(box.cls[0])
            status = "Con mastitis" if cls_id == 1 else "Sin mastitis"

            final_name = f"{animal_id}_{uuid.uuid4().hex[:8]}.jpg"
            final_path = f"{UPLOAD_DIR}/{final_name}"
            cv2.imwrite(final_path, img)

            x1, y1, x2, y2, box_w, box_h = box_data
            area_pct = (box_w * box_h) / (w * h) * 100

            results_data.append({
                "image_position": i,
                "filename": filename,
                "valid": True,
                "status": status,
                "mastitis_detected": cls_id == 1,
                "confidence": round(confidence * 100, 2),
                "image_path": f"/uploads/{final_name}",
                "box": {
                    "x1": round(x1,2),
                    "y1": round(y1,2),
                    "x2": round(x2,2),
                    "y2": round(y2,2),
                    "area_percentage": round(area_pct,2)
                }
            })

        except Exception as e:
            results_data.append(build_error(i, filename, str(e)))

    # 🔹 RESULTADO FINAL
    has_mastitis = any(r.get("mastitis_detected") for r in results_data)
    final_conf = max([r.get("confidence", 0) for r in results_data], default=0)

    return {
        "animal_id": animal_id,
        "status": "Con mastitis" if has_mastitis else "Sin mastitis",
        "mastitis_detected": has_mastitis,
        "confidence": final_conf,
        "analysis_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "details": results_data
    }


@main.get("/health")
def health():
    return {"status": "ok", "model": "YOLO"}