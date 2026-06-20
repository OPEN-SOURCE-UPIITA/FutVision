"""
RoboCup Multi-Object Tracker - Robots & Ball Pipeline

Este script fusiona la detección de robots (azules y rojos) con la detección 
del balón naranja, manteniendo la arquitectura de alto rendimiento:
HSV -> ByteTrack -> SAM 3.

Visualización:
    - Máscaras (SAM 3) para robots y balón (azul, rojo, naranja).
    - Bounding Boxes y Labels únicamente para los robots.
    - Estela dinámica y filtro anti-salto exclusivamente para el balón.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM
from collections import deque

# --- 1. CONFIGURACIÓN DE PUNTOS Y HOMOGRAFÍA ---
SOURCE_POINTS = np.float32([[20, 165], [590, 40], [592, 1052], [15, 985]])
CAMPO_W, CAMPO_H = 364, 486
TARGET_POINTS = np.float32([[0, 0], [CAMPO_W, 0], [CAMPO_W, CAMPO_H], [0, CAMPO_H]])
H_TRANS = cv2.getPerspectiveTransform(SOURCE_POINTS, TARGET_POINTS)

# Mapeo de Clases y Colores
CLASS_NAMES = {0: "azul", 1: "rojo", 2: "balon"}
COLORS_HEX  = {0: "#00b4d8", 1: "#ef233c", 2: "#ff9500"}

# Carga de modelos
sam_model = SAM("sam3.pt")
print("✅ SAM 3 Cargado")

# Configuración de ByteTrack multi-objeto
tracker = sv.ByteTrack(
    track_activation_threshold=0.15,       # Admite detecciones veloces con blur
    lost_track_buffer=90,                  # Mantiene ID de objetos ocluidos por 3 segundos a 30fps
    minimum_matching_threshold=0.4,        # Tolerancia a movimientos bruscos inter-frame
    frame_rate=30
)

# Inicializar Anotadores de Supervision
palette   = sv.ColorPalette.from_hex(list(COLORS_HEX.values()))
mask_ann  = sv.MaskAnnotator(color=palette, opacity=0.5)
box_ann   = sv.BoxAnnotator(color=palette, thickness=2)
label_ann = sv.LabelAnnotator(color=palette, text_color=sv.Color.WHITE)

# Configuración de la estela exclusiva del balón
MAX_TRAIL_LEN = 45
ball_trail = deque(maxlen=MAX_TRAIL_LEN)

# Filtro anti-salto para la estela del balón
last_confirmed_ball_pos = None   
MAX_JUMP_DISTANCE = 600  # Máxima distancia física lógica en píxeles por frame

# --- 2. DETECTOR MULTI-OBJETO (ROBOTS & BALÓN) ---
def detect_robots_and_ball_hsv(frame_bgr: np.ndarray, min_area: int = 15) -> sv.Detections:
    """
    Detecta robots (azul=0, rojo=1) y el balón (naranja=2) utilizando 
    máscaras HSV independientes y unificando el formato de salida para el Tracker.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Definición de Máscaras de Color ---
    # Marcador Cyan (Robots Azules)
    mask_azul = cv2.inRange(hsv, np.array([85, 180, 100]), np.array([105, 255, 255]))

    # Marcador Rojo (Robots Rojos - Rango doble para el canal de tono rojo en HSV)
    mask_rojo = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
    mask_rojo_high = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([179, 255, 255]))
    mask_rojo = cv2.bitwise_or(mask_rojo, mask_rojo_high)

    # Balón Naranja (Rango amplio compatible con motion blur)
    mask_balon = cv2.inRange(hsv, np.array([5, 180, 180]), np.array([30, 255, 255]))

    xyxy_list, class_ids, confidences = [], [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Lista de tuplas: (Máscara, Class_ID)
    targets = [(mask_azul, 0), (mask_rojo, 1), (mask_balon, 2)]

    for mask, cid in targets:
        # Limpieza morfológica por máscara
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > 5000: 
                continue

            x, y, w, h_box = cv2.boundingRect(cnt)

            # Filtros geométricos
            rect_area = w * h_box
            if rect_area == 0: 
                continue
            solidity = area / rect_area
            if solidity < 0.25: 
                continue

            aspect_ratio = float(w) / h_box
            if not (0.15 < aspect_ratio < 6.0): 
                continue

            # --- PADDING ADAPTATIVO POR CLASE ---
            # Los robots usan un marcador pequeño en su parte superior, por lo que 
            # necesitan expandir significativamente la caja para que SAM capture el chasis completo.
            if cid == 2:
                pad_w, pad_h = 20, 20  # Balón
            else:
                pad_w, pad_h = 40, 40  # Robots (Azul / Rojo)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame_bgr.shape[1], x + w + pad_w)
            y2 = min(frame_bgr.shape[0], y + h_box + pad_h)

            xyxy_list.append([x1, y1, x2, y2])
            class_ids.append(cid)
            confidences.append(min(1.0, 0.3 + 0.7 * solidity))

    if not xyxy_list: 
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confidences, dtype=np.float32)
    )

# --- 3. BUCLE DE PROCESAMIENTO DE VIDEO ---
video_path = "your_path"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: 
    fps = 30

out = cv2.VideoWriter('your_path', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret: 
        break

    # Detección HSV Multi-Objeto
    dets_hsv = detect_robots_and_ball_hsv(frame_bgr)
    
    # Actualización del Tracker
    dets_tracked = tracker.update_with_detections(dets_hsv)

    frame_out = frame_bgr.copy()

    if len(dets_tracked) > 0:
        # Segmentación Fina (SAM 3) usando las cajas de ByteTrack como Prompt
        bboxes = dets_tracked.xyxy.tolist()
        results = sam_model(frame_bgr, bboxes=bboxes, verbose=False)
        dets_sam = sv.Detections.from_ultralytics(results[0])
           
        # Sincronización Segura SAM + ByteTrack
        if dets_sam.mask is not None and len(dets_sam.mask) == len(dets_tracked):
            dets_limpio = sv.Detections(
                xyxy=dets_tracked.xyxy,
                mask=dets_sam.mask,
                class_id=dets_tracked.class_id,
                tracker_id=dets_tracked.tracker_id
            )
            
            # --- RENDERIZADO CONDICIONAL DE OBJETOS ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. Pintar MÁSCARAS para todos los elementos (clases 0, 1 y 2)
            frame_out = cv2.cvtColor(mask_ann.annotate(frame_rgb, dets_limpio), cv2.COLOR_RGB2BGR)
            
            # Separar Robots del Balón para aplicar anotadores condicionales
            is_robot = (dets_limpio.class_id == 0) | (dets_limpio.class_id == 1)
            is_ball  = (dets_limpio.class_id == 2)
            
            dets_robots = dets_limpio[is_robot]
            dets_ball   = dets_limpio[is_ball]
            
            # 2. Pintar Bounding Boxes y Etiquetas ÚNICAMENTE en los Robots (Azul/Rojo)
            if len(dets_robots) > 0:
                labels_robots = [
                    f"{CLASS_NAMES[cid]} #{tid}" 
                    for cid, tid in zip(dets_robots.class_id, dets_robots.tracker_id)
                ]
                frame_out = box_ann.annotate(frame_out, dets_robots)
                frame_out = label_ann.annotate(frame_out, dets_robots, labels=labels_robots)
            
            # --- 3. LOGICA DE ESTELA EXCLUSIVA DEL BALÓN ---
            if len(dets_ball) > 0:
                ball_box = dets_ball.xyxy[0]
                cx = int((ball_box[0] + ball_box[2]) / 2)
                cy = int((ball_box[1] + ball_box[3]) / 2)

                # Control anti-salto de estela
                if last_confirmed_ball_pos is not None:
                    distance = np.sqrt((cx - last_confirmed_ball_pos[0])**2 + (cy - last_confirmed_ball_pos[1])**2)
                    if distance <= MAX_JUMP_DISTANCE:
                        ball_trail.append((cx, cy))
                        last_confirmed_ball_pos = (cx, cy)
                    else:
                        ball_trail.append(None) # Romper línea diagonal errónea si salta por falso positivo
                else:
                    ball_trail.append((cx, cy))
                    last_confirmed_ball_pos = (cx, cy)
            else:
                ball_trail.append(None)
                # Auto-reset si se pierde por más de 10 cuadros consecutivos
                if len(ball_trail) > 10 and all(p is None for p in list(ball_trail)[-10:]):
                    last_confirmed_ball_pos = None
        else:
            # Fallback seguro si SAM no generó máscara sincronizada
            frame_out = frame_bgr.copy()
            ball_trail.append(None)
    else:
        ball_trail.append(None)

    # 4. Dibujar la Estela (Trail) del balón usando degradado dinámico (BGR)
    active_trail = [p for p in ball_trail if p is not None]
    for i in range(1, len(active_trail)):
        if ball_trail[i - 1] is None or ball_trail[i] is None: 
            continue
        thickness = int(np.sqrt(MAX_TRAIL_LEN / float(i)) * 2.5)
        cv2.line(frame_out, active_trail[i - 1], active_trail[i], (0, 149, 255), thickness)

    out.write(frame_out)
    frame_idx += 1

cap.release()
out.release()
print("🎉 Procesamiento de Robots y Balón finalizado con éxito!")
