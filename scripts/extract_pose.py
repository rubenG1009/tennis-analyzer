#!/usr/bin/env python3
"""Extrae landmarks de pose de un video de tenis a CSV largo + video de debug.

Usa exclusivamente la API mediapipe.tasks (PoseLandmarker). La API antigua
mp.solutions no existe en mediapipe 1.x, por eso los nombres de los 33
landmarks de BlazePose y las conexiones del esqueleto van hardcodeados aqui.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ObjectDetector,
    ObjectDetectorOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# Los 33 landmarks de BlazePose, en orden de indice.
NOMBRES_LANDMARKS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Huesos del esqueleto: pares de indices que se unen con una linea.
CONEXIONES = [
    # cara
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # brazo izquierdo
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # brazo derecho
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # pierna izquierda
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # pierna derecha
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Landmarks del lado derecho que interesan para analizar la tecnica del golpe.
LANDMARKS_INFORME = [
    "right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee",
]

UMBRAL_VISIBILITY = 0.5

# Deteccion de cortes de escena. Medidos sobre los clips de referencia: los
# cortes reales dan una diferencia media >= 20 (escala de grises 0-255 sobre una
# miniatura de 64x36) y el movimiento normal, incluidos barridos de camara,
# <= 15. El histograma de color cubre cortes a negro y rotulos.
UMBRAL_CORTE_DIFERENCIA = 18.0
UMBRAL_CORTE_HISTOGRAMA = 0.5

MAX_RAQUETAS = 3


def obtener_fourcc(codec):
    """OpenCV 5 pudo mover VideoWriter_fourcc a VideoWriter.fourcc."""
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter.fourcc(*codec)


def calcular_t_ms(idx, fps, ultimo_t_ms):
    """Timestamp en ms del frame idx, garantizando monotonia estricta.

    Funcion pura para poder testearla sin video. mediapipe exige timestamps
    estrictamente crecientes; con fps muy altos el redondeo puede repetir
    valor, asi que forzamos el avance minimo de 1 ms.
    """
    t_ms = round(idx * 1000 / fps)
    if t_ms <= ultimo_t_ms:
        t_ms = ultimo_t_ms + 1
    return t_ms


def calcular_estadisticas_visibility(valores, total_frames, umbral=None):
    """Media, minima y % de valores >= umbral de una serie de visibility.

    Funcion pura. `valores` son solo los frames CON deteccion; el porcentaje
    se calcula sobre `total_frames` (todos los leidos), porque un frame sin
    deteccion tampoco alcanza el umbral. Sin valores devuelve
    (None, None, 0.0) para que el informe lo pinte como "-".

    El criterio es >= , el mismo que usa dibujar_esqueleto para pintar verde:
    ambos leen UMBRAL_VISIBILITY, asi que no pueden desincronizarse.
    """
    if umbral is None:
        umbral = UMBRAL_VISIBILITY
    if not valores:
        return None, None, 0.0
    media = float(np.mean(valores))
    minima = float(np.min(valores))
    n_sobre = sum(1 for v in valores if v >= umbral)
    pct = 100 * n_sobre / total_frames if total_frames else 0.0
    return media, minima, pct


def firma_frame(frame):
    """Miniatura en grises e histograma H-S de un frame, para comparar frames."""
    pequeno = cv2.resize(frame, (64, 36))
    gris = cv2.GaussianBlur(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return gris.astype(np.float32), cv2.normalize(hist, hist).flatten()


def es_corte(firma_prev, firma):
    """True si entre los dos frames hay un corte de escena. Funcion pura."""
    if firma_prev is None:
        return False
    diferencia = float(np.mean(np.abs(firma[0] - firma_prev[0])))
    correlacion = cv2.compareHist(firma_prev[1], firma[1], cv2.HISTCMP_CORREL)
    return diferencia > UMBRAL_CORTE_DIFERENCIA or correlacion < UMBRAL_CORTE_HISTOGRAMA


def dibujar_esqueleto(frame, landmarks, ancho, alto):
    """Pinta huesos y articulaciones sobre el frame (coordenadas normalizadas)."""
    # Pixeles enteros de cada landmark, para no recalcularlos en cada hueso.
    puntos = [(int(lm.x * ancho), int(lm.y * alto)) for lm in landmarks]

    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], (255, 255, 255), 2)

    for i, (px, py) in enumerate(puntos):
        # Verde si el landmark es fiable, rojo si mediapipe no lo ve claro.
        color = (0, 255, 0) if landmarks[i].visibility >= UMBRAL_VISIBILITY else (0, 0, 255)
        cv2.circle(frame, (px, py), 4, color, -1)


def procesar_video(ruta_video, ruta_modelo, out_dir, ruta_modelo_raqueta=None):
    """Procesa el video entero: escribe el CSV largo, el video de debug
    e imprime el informe. Devuelve las rutas generadas y el resumen.

    En cada corte de escena se recrea el PoseLandmarker. En modo VIDEO el
    modelo sigue la region del frame anterior en vez de volver a detectar, y
    tras un corte se queda enganchado a quien este en esa zona (en los clips de
    referencia, un jugador del fondo durante 35 frames). Recrearlo fuerza una
    deteccion nueva. Los cortes se escriben en <nombre>_cortes.csv.

    Con ruta_modelo_raqueta se detecta ademas la raqueta en cada frame (clase
    COCO "tennis racket") y se escribe <nombre>_raqueta.csv; corregir_esqueleto
    la usa para saber cual es la mano dominante.

    Separada de main() para poder invocar el pipeline desde los tests sin
    pasar por el parseo de argumentos.
    """
    os.makedirs(out_dir, exist_ok=True)

    captura = cv2.VideoCapture(ruta_video)
    if not captura.isOpened():
        sys.exit(f"ERROR: OpenCV no puede abrir {ruta_video}")

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    frames_declarados = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        captura.release()
        sys.exit("ERROR: el video declara fps <= 0, no se puede calcular el eje temporal")

    base = os.path.splitext(os.path.basename(ruta_video))[0]
    ruta_csv = os.path.join(out_dir, f"{base}_landmarks.csv")
    ruta_debug = os.path.join(out_dir, f"{base}_debug.mp4")
    ruta_cortes = os.path.join(out_dir, f"{base}_cortes.csv")
    ruta_raqueta = os.path.join(out_dir, f"{base}_raqueta.csv")

    escritor_video = cv2.VideoWriter(ruta_debug, obtener_fourcc("mp4v"), fps, (ancho, alto))
    if not escritor_video.isOpened():
        captura.release()
        sys.exit(f"ERROR: no se puede escribir el video de debug en {ruta_debug}")

    opciones = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ruta_modelo),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    detector_raqueta = None
    f_raqueta = None
    if ruta_modelo_raqueta:
        detector_raqueta = ObjectDetector.create_from_options(ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=ruta_modelo_raqueta),
            running_mode=RunningMode.IMAGE,
            score_threshold=0.2,
            max_results=MAX_RAQUETAS,
            category_allowlist=["tennis racket"],
        ))
        f_raqueta = open(ruta_raqueta, "w", newline="")
        escritor_raqueta = csv.writer(f_raqueta)
        escritor_raqueta.writerow(["frame", "score", "x0", "y0", "x1", "y1"])

    frames_leidos = 0
    frames_sin_deteccion = 0
    cortes = []
    firma_prev = None
    ultimo_t_ms = -1
    # visibility acumulada por landmark de interes, solo de frames con deteccion.
    visibilidades = {nombre: [] for nombre in LANDMARKS_INFORME}

    print(f"Procesando {ruta_video} ...")

    landmarker = PoseLandmarker.create_from_options(opciones)
    with open(ruta_csv, "w", newline="") as f_csv:

        escritor_csv = csv.writer(f_csv)
        escritor_csv.writerow(
            ["frame", "t_ms", "landmark_id", "landmark", "x", "y", "z", "visibility"]
        )

        while True:
            ok, frame = captura.read()
            if not ok:
                break

            idx = frames_leidos
            t_ms = calcular_t_ms(idx, fps, ultimo_t_ms)
            ultimo_t_ms = t_ms

            firma = firma_frame(frame)
            if es_corte(firma_prev, firma):
                cortes.append(idx)
                landmarker.close()
                landmarker = PoseLandmarker.create_from_options(opciones)
            firma_prev = firma

            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            resultado = landmarker.detect_for_video(imagen_mp, t_ms)

            if detector_raqueta is not None:
                for d in detector_raqueta.detect(imagen_mp).detections:
                    b = d.bounding_box
                    escritor_raqueta.writerow([
                        idx, f"{d.categories[0].score:.3f}",
                        f"{b.origin_x / ancho:.5f}", f"{b.origin_y / alto:.5f}",
                        f"{(b.origin_x + b.width) / ancho:.5f}",
                        f"{(b.origin_y + b.height) / alto:.5f}",
                    ])
                    x0, y0 = b.origin_x, b.origin_y
                    cv2.rectangle(frame, (x0, y0), (x0 + b.width, y0 + b.height),
                                  (255, 128, 0), 2)

            if resultado.pose_landmarks:
                landmarks = resultado.pose_landmarks[0]
                for i, lm in enumerate(landmarks):
                    escritor_csv.writerow(
                        [idx, t_ms, i, NOMBRES_LANDMARKS[i],
                         f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}",
                         f"{lm.visibility:.6f}"]
                    )
                    nombre = NOMBRES_LANDMARKS[i]
                    if nombre in visibilidades:
                        visibilidades[nombre].append(lm.visibility)
                dibujar_esqueleto(frame, landmarks, ancho, alto)
            else:
                frames_sin_deteccion += 1
                # Escribimos las 33 filas vacias igualmente: el eje temporal no
                # puede tener huecos silenciosos aguas abajo.
                for i, nombre in enumerate(NOMBRES_LANDMARKS):
                    escritor_csv.writerow([idx, t_ms, i, nombre, "", "", "", ""])

            cv2.putText(frame, f"frame {idx}  t {t_ms} ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            escritor_video.write(frame)

            frames_leidos += 1
            if frames_leidos % 50 == 0:
                print(f"  {frames_leidos} frames procesados...")

    landmarker.close()
    if detector_raqueta is not None:
        detector_raqueta.close()
        f_raqueta.close()
    captura.release()
    escritor_video.release()
    with open(ruta_cortes, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["frame"])
        escritor.writerows([c] for c in cortes)

    # ----- informe -----
    duracion_s = frames_leidos / fps
    pct_sin = 100 * frames_sin_deteccion / frames_leidos if frames_leidos else 0.0

    print()
    print("=" * 62)
    print("INFORME DE EXTRACCION")
    print("=" * 62)
    print(f"Video              : {ruta_video}")
    print(f"Modelo             : {ruta_modelo}")
    print(f"Resolucion         : {ancho} x {alto}")
    print(f"FPS                : {fps:.3f}")
    print(f"Frames declarados  : {frames_declarados}")
    print(f"Frames leidos      : {frames_leidos}")
    print(f"Duracion           : {duracion_s:.2f} s")
    print(f"Frames sin deteccion: {frames_sin_deteccion} ({pct_sin:.2f} %)")
    print(f"Cortes de escena   : {len(cortes)} {cortes[:20]}")
    print()
    print("Visibility de landmarks clave del lado derecho")
    print("(media y minima sobre frames CON deteccion; % sobre el total de frames leidos)")
    rotulo_pct = f"% >= {UMBRAL_VISIBILITY}"
    print(f"{'landmark':<16}{'media':>9}{'minima':>9}{rotulo_pct:>10}")
    for nombre in LANDMARKS_INFORME:
        media, minima, pct = calcular_estadisticas_visibility(
            visibilidades[nombre], frames_leidos)
        if media is None:
            print(f"{nombre:<16}{'-':>9}{'-':>9}{'-':>10}")
            continue
        print(f"{nombre:<16}{media:>9.3f}{minima:>9.3f}{pct:>9.2f}%")
    print()
    print(f"CSV   -> {ruta_csv}")
    print(f"Debug -> {ruta_debug}")
    print("=" * 62)

    return {
        "ruta_csv": ruta_csv,
        "ruta_debug": ruta_debug,
        "frames_leidos": frames_leidos,
        "frames_sin_deteccion": frames_sin_deteccion,
        "cortes": cortes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extrae landmarks de pose de un video a CSV largo y video de debug."
    )
    parser.add_argument("video", help="Ruta del video de entrada")
    parser.add_argument("--model", default="models/pose_landmarker_heavy.task",
                        help="Ruta del .task de PoseLandmarker")
    parser.add_argument("--out-dir", default="data/processed",
                        help="Carpeta donde escribir el CSV y el video de debug")
    parser.add_argument("--raqueta", metavar="MODELO",
                        help="ObjectDetector COCO (p.ej. models/efficientdet_lite2.tflite) "
                             "para detectar la raqueta en cada frame")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"ERROR: no existe el video {args.video}")
    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: no existe el modelo {args.model}")

    if args.raqueta and not os.path.isfile(args.raqueta):
        sys.exit(f"ERROR: no existe el modelo {args.raqueta}")

    procesar_video(args.video, args.model, args.out_dir, args.raqueta)


if __name__ == "__main__":
    main()
