"""
RoboCup Autonomous Robot Tracker - Pipeline de Detección Dinámica y Segmentación

Este script procesa video de partidos de RoboCup de manera autónoma sin necesidad de 
calibrar manualmente la ubicación de la cancha. Utiliza visión por computadora clásica 
para aislar el área de juego y detectar robots, aplicando posteriormente ByteTrack para 
el rastreo inter-frame y SAM 3 para el refinamiento pixel-level de sus máscaras.

Flujo de Trabajo:
    1. Descarga de recursos (modelo SAM 3 y video de prueba).
    2. Detección autónoma del contorno y límites de la cancha verde mediante HSV.
    3. Segmentación y filtrado de candidatos de robots excluyendo elementos inválidos.
    4. Seguimiento temporal continuo de robots (ByteTrack).
    5. Segmentación fina sobre las cajas de interés (SAM 3).
    6. Anotación visual con degradados e IDs persistentes.
"""

import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM
from trackers import ByteTrackTracker  # Asegúrate de tener este módulo local listo
from pathlib import Path
import torch
import urllib.request

# ============================================================================
# 1. DESCARGA AUTOMÁTICA DE RECURSOS (ASSETS)
# ============================================================================

ASSETS_BASE = "https://futbot-eight.vercel.app/assets"
VIDEO_URL   = "https://docs.google.com/videos/d/1jw99aFlfR5pzjjlEC9jFyjH20O1a656ETmiQm7BLtZE/edit?scene=id.g5bdee242_0_2#scene=id.g5bdee242_0_2"

ASSETS = [
    "sam3.pt",
]

# Creamos la carpeta local para almacenar los recursos del proyecto si no existe
Path("assets").mkdir(exist_ok=True)

# Descargamos los modelos preentrenados del servidor del curso
for name in ASSETS:
    dest = Path("assets") / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"📦 {name} ya existe en el directorio local.")
    else:
        print(f"📥 Descargando {name}...")
        urllib.request.urlretrieve(f"{ASSETS_BASE}/{name}", dest)
        print(f"💾 Guardado con éxito en {dest}")

# Descargamos el video de prueba asociado
video_dest = Path("assets/vidio8.mov")
if video_dest.exists() and video_dest.stat().st_size > 0:
    print(f"📦 {video_dest.name} ya existe en el directorio local.")
else:
    print("📥 Descargando video de prueba...")
    urllib.request.urlretrieve(VIDEO_URL, video_dest)
    print(f"💾 Guardado con éxito en {video_dest}")

print("✨ Todos los recursos están listos para la ejecución.")


# ============================================================================
# 2. CONFIGURACIÓN DE DISPOSITIVOS, PARÁMETROS Y MODELOS
# ============================================================================

VIDEO_INPUT  = "assets/vidio8.mov"
VIDEO_OUTPUT = "assets/output_tracking.mp4"
SAM_MODEL    = "assets/sam3.pt"

# Paleta de colores RGB/BGR para distinguir los IDs de los robots trackeados
COLORES_TRACK = [
    (255, 100,   0),  # Naranja
    (  0, 200, 255),  # Cyan
    (100, 255,   0),  # Verde lima
    (255,   0, 180),  # Magenta
    (  0, 255, 150),  # Esmeralda
    (200,   0, 255),  # Violeta
]

# Selección automática de aceleración por hardware (CUDA para GPUs de NVIDIA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Dispositivo de cómputo seleccionado: {device}")
if device == "cuda":
    print(f"🚀 GPU Detectada: {torch.cuda.get_device_name(0)}")

# Carga de SAM3 en la memoria del dispositivo configurado (CPU/GPU)
sam_model = SAM(SAM_MODEL)
sam_model.to(device)

# Inicializamos el tracker de robots personalizado
tracker = ByteTrackTracker()

# --- NUEVO: Inicialización robusta de Anotadores de Supervision ---
# Mapea tu paleta de tuplas RGB/BGR personalizadas al formato Color de Supervision
palette = sv.ColorPalette([sv.Color(*c) for c in COLORES_TRACK])

mask_annotator = sv.MaskAnnotator(color=palette, opacity=0.5)
box_annotator = sv.BoxAnnotator(color=palette, thickness=2)
label_annotator = sv.LabelAnnotator(color=palette, text_color=sv.Color.WHITE)
trace_annotator = sv.TraceAnnotator(color=palette, thickness=2)


# ============================================================================
# 3. DETECTOR GEOMÉTRICO AUTÓNOMO DE LA CANCHA
# ============================================================================

def detectar_cancha(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detecta de forma autónoma las 4 esquinas de la cancha verde usando segmentación HSV.
    
    Aplica operaciones morfológicas robustas y aproximación poligonal para obtener
    los límites precisos del campo de juego en perspectiva, ignorando espectadores o entornos.

    Args:
        frame_bgr (np.ndarray): Frame original en formato BGR.

    Returns:
        np.ndarray | None: Array float32 de coordenadas de esquinas (4, 2), o None si falla.
    """
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Rango de color verde adaptado para el tapete del campo de juego
    mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([90, 255, 255]))

    # Limpieza morfológica agresiva (15x15) para unificar la cancha y eliminar líneas o marcas internas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Seleccionamos el contorno de mayor tamaño (que necesariamente será la cancha verde)
    cancha_cnt = max(contours, key=cv2.contourArea)

    # Validamos que el contorno ocupe al menos el 10% del total de la imagen para descartar falsas alarmas
    area_frame = frame_bgr.shape[0] * frame_bgr.shape[1]
    if cv2.contourArea(cancha_cnt) < area_frame * 0.10:
        return None

    # Aproximación polinómica para encontrar las 4 esquinas del trapecio en perspectiva
    epsilon = 0.01 * cv2.arcLength(cancha_cnt, True)
    approx  = cv2.approxPolyDP(cancha_cnt, epsilon, True)

    # Si por oclusiones el trapecio no arroja 4 esquinas exactas, calculamos su rectángulo mínimo rotado
    if len(approx) != 4:
        rect   = cv2.minAreaRect(cancha_cnt)
        box    = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.float32)

    return approx.reshape(-1, 2).astype(np.float32)


# ============================================================================
# 4. DETECTOR DE ROBOTS CANDIDATOS (SUSTRACCIÓN DE COLOR EN CANCHA)
# ============================================================================

def detectar_robots_candidatos(frame: np.ndarray, esquinas_cancha: np.ndarray) -> list:
    """
    Identifica de manera robusta las cajas de robots dentro del área de juego.
    
    Genera una máscara para aislar el campo de juego aplicando erosión interna para 
    evitar que las bandas de madera negra distorsionen los contornos. Posteriormente, 
    elimina selectivamente el tapete verde, las líneas blancas y el balón naranja, dejando 
    como candidatos únicamente los chasis de los robots.

    Args:
        frame (np.ndarray): Frame original BGR.
        esquinas_cancha (np.ndarray): Coordenadas (4, 2) del límite de la cancha.

    Returns:
        list: Lista con las cajas de robots detectadas [x1, y1, x2, y2].
    """
    # 1. Crear máscara de la cancha y erosionarla levemente para evitar el borde de madera negro
    cancha_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(cancha_mask, esquinas_cancha.astype(np.int32), 255)
    cancha_mask = cv2.erode(
        cancha_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Definición de umbrales HSV de exclusión
    green_mask  = cv2.inRange(hsv, (35,  40,  40), (95,  255, 255))   # Tapete verde
    white_mask  = cv2.inRange(hsv, (0,   0,  170), (180,  80, 255))   # Líneas blancas de cal / Robots blancos
    orange_mask = cv2.inRange(hsv, (5,  80,   80), (25,  255, 255))   # Balón naranja

    # 3. Sustracción de elementos: Excluimos verde, líneas blancas y balón dentro de la cancha erosionada
    objetos = cv2.bitwise_not(green_mask)
    objetos = cv2.bitwise_and(objetos, cancha_mask)
    objetos = cv2.bitwise_and(objetos, cv2.bitwise_not(white_mask))
    objetos = cv2.bitwise_and(objetos, cv2.bitwise_not(orange_mask))

    # Limpieza morfológica de los chasis candidatos para unificar contornos irregulares
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    objetos = cv2.morphologyEx(objetos, cv2.MORPH_OPEN,  kernel)
    objetos = cv2.morphologyEx(objetos, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filtro de área extrema en píxeles para ignorar motas de polvo o distorsiones masivas
        if area < 1500 or area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtro de Aspect Ratio para asegurar que la caja tenga proporciones compatibles con robots
        aspect = w / float(h)
        if aspect < 0.45 or aspect > 2.20:
            continue

        # Filtro de Solidez (Fill Ratio) para confirmar que no sea una línea delgada o hueco vacío
        roi        = objetos[y:y+h, x:x+w]
        fill_ratio = cv2.countNonZero(roi) / max((w * h), 1)
        if fill_ratio < 0.25 or fill_ratio > 0.90:
            continue

        # Caja válida de candidato de robot detectada
        boxes.append([x, y, x + w, y + h])

    return boxes


# ============================================================================
# 5. PIPELINE DE PROCESAMIENTO UNITARIO DE FRAMES
# ============================================================================

def procesar_frame(frame: np.ndarray) -> np.ndarray:
    """
    Ejecuta el pipeline unificado sobre un único frame:
    Esquinas Cancha -> Candidatos Robots -> Tracking (ByteTrack) -> SAM3 -> Renderizado.

    Args:
        frame (np.ndarray): Imagen actual en formato BGR.

    Returns:
        np.ndarray: Imagen BGR resultante con overlays gráficos de tracking.
    """
    # 1. Detectamos autónomamente la cancha en este frame
    esquinas = detectar_cancha(frame)
    if esquinas is None:
        return frame

    # 2. Localizamos candidatos de robots en base al área de juego
    boxes = detectar_robots_candidatos(frame, esquinas)
    if len(boxes) == 0:
        return frame

    # Convertimos los candidatos detectados al formato estándar sv.Detections
    detections = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.ones(len(boxes), dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=np.int32)
    )

    # Ventana de depuración interactiva OpenCV para monitorizar los candidatos sin procesar
    debug = frame.copy()
    for box in detections.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(debug, (0, 255, 0), (x1, y1), (x2, y2), 2)
    cv2.imshow("DEBUG ROBOTS", debug)
    cv2.waitKey(1)

    # 3. Pasamos las detecciones al tracker para heredar IDs persistentes inter-frame
    detections = tracker.update(detections)

    sam_xyxy_list = []
    sam_mask_list = []
    sam_tid_list  = []
    sam_cid_list  = []
    sam_conf_list = []

    # 4. Solicitamos inferencia fina de SAM 3 sobre cada Bounding Box trackeado individualmente
    for box, tid, cid, conf in zip(
            detections.xyxy, detections.tracker_id,
            detections.class_id, detections.confidence):
        try:
            sam_result = sam_model(frame, bboxes=[box.tolist()], verbose=False)[0]
            sam_det    = sv.Detections.from_ultralytics(sam_result)
            if len(sam_det) > 0:
                sam_xyxy_list.append(sam_det.xyxy[0])
                if sam_det.mask is not None:
                    sam_mask_list.append(sam_det.mask[0])
                sam_tid_list.append(tid)
                sam_cid_list.append(cid)
                sam_conf_list.append(conf)
        except Exception:
            # En caso de error o fallo del modelo, usamos la caja original del tracker como fallback
            sam_xyxy_list.append(box)
            sam_tid_list.append(tid)
            sam_cid_list.append(cid)
            sam_conf_list.append(conf)

    if not sam_xyxy_list:
        return frame

    # Construimos un objeto unificado con las máscaras finas y los IDs persistentes
    sam_dets = sv.Detections(
        xyxy=np.array(sam_xyxy_list, dtype=np.float32),
        mask=np.array(sam_mask_list) if sam_mask_list else None,
        tracker_id=np.array(sam_tid_list, dtype=int),
        class_id=np.array(sam_cid_list, dtype=int),
        confidence=np.array(sam_conf_list, dtype=np.float32),
    )

    # 5. Renderizado final utilizando la configuración de anotadores
    annotated = frame.copy()
    annotated = mask_annotator.annotate(scene=annotated, detections=sam_dets)
    annotated = box_annotator.annotate(scene=annotated, detections=sam_dets)
    
    if sam_dets.tracker_id is not None:
        labels    = [f"Robot ID:{tid}" for tid in sam_dets.tracker_id]
        annotated = label_annotator.annotate(scene=annotated, detections=sam_dets, labels=labels)
        annotated = trace_annotator.annotate(scene=annotated, detections=sam_dets)

    return annotated


# ============================================================================
# 6. FUNCIÓN DE PROCESAMIENTO INTEGRAL DE VIDEO (E/S)
# ============================================================================

def procesar_video(input_path: str, output_path: str) -> None:
    """
    Lee y escribe video frame por frame ejecutando el pipeline de rastreo.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error crítico: No se pudo abrir el archivo de entrada: {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Códec mp4v estándar de OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w_orig, h_orig))

    print(f"Propiedades del video: {w_orig}x{h_orig} px | {fps:.1f} FPS | {total} frames")
    print(f"Destino de salida: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultado = procesar_frame(frame)
        writer.write(resultado)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f" ⏳ Progreso: {frame_idx}/{total} frames ({pct:.1f}%)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"🎉 Procesamiento de video completo. Archivo guardado con éxito en: {output_path}")

# Ejecutar el procesamiento de video completo
procesar_video(VIDEO_INPUT, VIDEO_OUTPUT)
