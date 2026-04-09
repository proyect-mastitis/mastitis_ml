from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
import shutil, os, uuid
import cv2
import numpy as np

app = FastAPI(title="Servidor de Análisis de Mastitis")


# 🔒 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000", "http://192.168.18.82:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

model = YOLO("models/best.pt")

# 📏 VALIDADORES DE CALIDAD
MIN_CONFIDENCE = 0.55
MIN_BOX_SIZE = 0.08
MAX_BOX_SIZE = 0.60
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 2.0

def validate_image_quality(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"valid": False, "error": "No se pudo leer la imagen. Intenta con otra foto."}

        height, width = img.shape[:2]
        if width < 320 or height < 320:
            return {"valid": False, "error": "La imagen no es clara. Toma la foto nuevamente."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 210:
            return {"valid": False, "error": "Problema de iluminación."}

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        if variance < 150:
            return {"valid": False, "error": "Imagen borrosa - necesita ser más nítida"}

        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Error al validar imagen: {str(e)}"}

def validate_udder_detection(box, img_width: float, img_height: float) -> dict:
    try:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = (box_width * box_height) / (img_width * img_height)

        if box_area < MIN_BOX_SIZE:
            return {"valid": False, "error": "Detección muy pequeña. Acerca más la cámara."}
        if box_area > MAX_BOX_SIZE:
            return {"valid": False, "error": "Detección muy grande. Probablemente no sea una ubre."}

        aspect_ratio = box_height / box_width
        if aspect_ratio < MIN_ASPECT_RATIO:
            return {"valid": False, "error": "Ubre muy ancha. Verifica el ángulo."}
        if aspect_ratio > MAX_ASPECT_RATIO:
            return {"valid": False, "error": "Ubre muy alta. Probablemente sea otro objeto."}

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x_ratio = center_x / img_width
        center_y_ratio = center_y / img_height
        if center_x_ratio < 0.1 or center_x_ratio > 0.9:
            return {"valid": False, "error": "Ubre detectada muy al borde. Centra la imagen."}
        if center_y_ratio < 0.2 or center_y_ratio > 0.9:
            return {"valid": False, "error": "Ubre detectada en posición incorrecta. Recaptura la imagen."}

        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Error validando ubre: {str(e)}"}

@app.post("/analyze")
async def analyze(animal_id: str = Form(...), files: list[UploadFile] = File(...)):
    if not animal_id:
        raise HTTPException(status_code=400, detail="animal_id es requerido")
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Debe subir al menos una imagen")
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Máximo 2 imágenes permitidas")

    invalid_images = []
    valid_paths = []
    file_index = 0

    # 🔹 Pre-validación de calidad antes de YOLO
    for file in files:
        file_index += 1
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            invalid_images.append({
                "image_position": file_index,
                "filename": file.filename,
                "valid": False,
                "error": "Formato no soportado",
                "image_path": None,
                "box": None,
                "confidence": 0
            })
            continue

        unique_name = f"uploads/temp_{uuid.uuid4()}{ext}"
        with open(unique_name, "wb") as f:
            shutil.copyfileobj(file.file, f)

        quality_check = validate_image_quality(unique_name)
        if not quality_check["valid"]:
            invalid_images.append({
                "image_position": file_index,
                "filename": file.filename,
                "valid": False,
                "error": quality_check["error"],
                "image_path": None,
                "box": None,
                "confidence": 0
            })
            try:
                os.remove(unique_name)
            except:
                pass
        else:
            valid_paths.append({"file": file, "path": unique_name, "position": file_index})

    if invalid_images:
        for vp in valid_paths:
            try:
                os.remove(vp["path"])
            except:
                pass
        raise HTTPException(
            status_code=400,
            detail={
                "message": "❌ Al menos una imagen no es válida",
                "reason": "Verifica que todas las imágenes sean claras y centradas de la ubre",
                "details": invalid_images
            }
        )

    results_data = []
    valid_count = 0

    # 🔹 Análisis YOLO
    for vp in valid_paths:
        file = vp["file"]
        path = vp["path"]
        position = vp["position"]

        try:
            results = model.predict(path, save=False, conf=0.3, verbose=False)

            if not results[0].boxes or len(results[0].boxes) == 0:
                results_data.append({
                    "image_position": position,
                    "filename": file.filename,
                    "valid": False,
                    "error": "❌ No se detecta ubre.",
                    "image_path": None,
                    "box": None,
                    "confidence": 0
                })
                continue

            confs = results[0].boxes.conf
            best_box_idx = int(confs.argmax())
            box = results[0].boxes[best_box_idx]
            cls_id = int(box.cls[best_box_idx].item())
            confidence = float(box.conf[best_box_idx].item())

            if confidence < MIN_CONFIDENCE:
                results_data.append({
                    "image_position": position,
                    "filename": file.filename,
                    "valid": False,
                    "error": f"Detección de baja confianza ({confidence*100:.1f}%).",
                    "image_path": None,
                    "box": None,
                    "confidence": round(confidence * 100, 2)
                })
                continue

            img = cv2.imread(path)
            img_height, img_width = img.shape[:2]
            box_validation = validate_udder_detection(box, img_width, img_height)

            if not box_validation["valid"]:
                results_data.append({
                    "image_position": position,
                    "filename": file.filename,
                    "valid": False,
                    "error": box_validation["error"],
                    "image_path": None,
                    "box": None,
                    "confidence": round(confidence * 100, 2)
                })
                continue

            # ✅ Imagen válida - GUARDAR
            valid_count += 1
            status = "Con mastitis" if cls_id == 1 else "Sin mastitis"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_image_name = f"analysis_{animal_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            final_image_path = os.path.join("uploads", final_image_name)
            shutil.copy(path, final_image_path)

            x1, y1, x2, y2 = map(float, box.xyxy[best_box_idx].tolist())
            box_width = x2 - x1
            box_height = y2 - y1
            box_area_pct = (box_width * box_height) / (img_width * img_height) * 100

            results_data.append({
                "image_position": position,
                "filename": file.filename,
                "valid": True,
                "status": status,
                "mastitis_detected": cls_id == 1,
                "confidence": round(confidence * 100, 2),
                "image_path": f"/uploads/{final_image_name}",
                "image_id": str(uuid.uuid4()),
                "image_width": img_width,
                "image_height": img_height,
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(box_width, 2),
                    "height": round(box_height, 2),
                    "area_percentage": round(box_area_pct, 2),
                    "center_x": round((x1 + x2) / 2, 2),
                    "center_y": round((y1 + y2) / 2, 2),
                }
            })

        except Exception as e:
            results_data.append({
                "image_position": position,
                "filename": file.filename,
                "valid": False,
                "error": f"Error procesando: {str(e)}",
                "image_path": None,
                "box": None,
                "confidence": 0
            })

    # ✅ Si todas son válidas
    has_mastitis = any(r["mastitis_detected"] for r in results_data)
    final_status = "Con mastitis" if has_mastitis else "Sin mastitis"
    final_conf = round(max(r["confidence"] for r in results_data), 2)

    # 🔹 Limpiar temporales
    for vp in valid_paths:
        try:
            if os.path.exists(vp["path"]):
                os.remove(vp["path"])
        except:
            pass

    return {
        "animal_id": animal_id,
        "status": final_status,
        "mastitis_detected": has_mastitis,
        "confidence": final_conf,
        "analysis_date": datetime.now().strftime("%d/%m/%Y, %H:%M"),
        "valid_count": valid_count,
        "total_uploaded": len(files),
        "is_valid": True,
        "details": results_data
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "model": "YOLO", "uploads_folder": os.path.exists("uploads")}