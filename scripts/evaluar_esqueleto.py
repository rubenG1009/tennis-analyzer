#!/usr/bin/env python3
"""Mide la calidad del esqueleto de un CSV de landmarks sin anotaciones manuales.

No hay ground truth, asi que se buscan sintomas de error que se pueden medir
con la propia serie temporal:

- picos: un landmark salta lejos un frame y vuelve al siguiente. El movimiento
  real, por rapido que sea, es suave en 3 frames; un pico aislado es un fallo
  de colocacion.
- cruces: izquierda y derecha intercambiadas de un frame al siguiente (muneca
  izquierda donde estaba la derecha y viceversa). Es el error tipico de
  BlazePose en vista lateral, donde un brazo tapa al otro.
- saltos de persona: el centro de la cadera se teletransporta o la escala del
  torso cambia de golpe. Indica que el modelo ha saltado a otra persona
  (rival, entrenador, recogepelotas) o que hay un corte de escena.
- visibility baja en el brazo que golpea.

Todas las distancias se expresan en longitudes de torso (hombro medio a cadera
media, en pixeles), para que sean comparables entre clips de distinta
resolucion y encuadre.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

UMBRAL_VISIBILITY = 0.5
UMBRAL_PICO = 0.35          # torsos de desviacion respecto al punto medio de sus vecinos
UMBRAL_SALTO_CADERA = 0.6   # torsos que se mueve el centro de cadera en un frame
UMBRAL_CAMBIO_ESCALA = 0.35 # cambio relativo del torso en un frame
RATIO_CRUCE = 0.5           # el emparejamiento cruzado cuesta menos de la mitad que el directo
MIN_MOV_CRUCE = 0.15        # en torsos; por debajo no se evalua cruce (ruido)

PARES_LR = ["wrist", "elbow", "knee", "ankle"]
HUESOS = {
    "brazo": ("shoulder", "elbow"),
    "antebrazo": ("elbow", "wrist"),
    "muslo": ("hip", "knee"),
    "tibia": ("knee", "ankle"),
}


def cargar(ruta_csv, ancho, alto):
    """CSV largo -> dict nombre_landmark -> array (n_frames, 2) en pixeles, y visibility.

    Los frames sin deteccion quedan como NaN.
    """
    df = pd.read_csv(ruta_csv)
    n = int(df["frame"].max()) + 1
    xy = {}
    vis = {}
    for nombre, g in df.groupby("landmark"):
        g = g.set_index("frame").reindex(range(n))
        xy[nombre] = np.column_stack([g["x"].to_numpy() * ancho, g["y"].to_numpy() * alto])
        vis[nombre] = g["visibility"].to_numpy()
    return n, xy, vis


def punto_medio(a, b):
    return (a + b) / 2


def longitud_torso(xy):
    return np.linalg.norm(
        punto_medio(xy["left_shoulder"], xy["right_shoulder"])
        - punto_medio(xy["left_hip"], xy["right_hip"]), axis=1)


def detectar_picos(serie, torso, umbral=UMBRAL_PICO):
    """Frames t en los que serie[t] se aleja del punto medio de t-1 y t+1.

    Funcion pura. `serie` es (n, 2) en pixeles; devuelve un array booleano (n,).
    """
    n = len(serie)
    picos = np.zeros(n, dtype=bool)
    if n < 3:
        return picos
    medio = (serie[:-2] + serie[2:]) / 2
    desv = np.zeros(n)
    desv[1:-1] = np.nan_to_num(np.linalg.norm(serie[1:-1] - medio, axis=1) / torso[1:-1], nan=0.0)
    # Un pico de altura h hace que sus dos vecinos se desvien h/2: solo cuenta
    # el maximo local, o se "corregirian" tambien los vecinos buenos.
    maximo_local = np.zeros(n, dtype=bool)
    maximo_local[1:-1] = (desv[1:-1] >= desv[:-2]) & (desv[1:-1] >= desv[2:])
    return (desv > umbral) & maximo_local


def detectar_cruces(izq, der, torso, ratio=RATIO_CRUCE, min_mov=MIN_MOV_CRUCE):
    """Frames t en los que izq/der encajan mejor cruzados con t-1 que directos.

    Funcion pura. Devuelve un array booleano (n,).
    """
    n = len(izq)
    cruces = np.zeros(n, dtype=bool)
    if n < 2:
        return cruces
    directo = (np.linalg.norm(izq[1:] - izq[:-1], axis=1)
               + np.linalg.norm(der[1:] - der[:-1], axis=1))
    cruzado = (np.linalg.norm(izq[1:] - der[:-1], axis=1)
               + np.linalg.norm(der[1:] - izq[:-1], axis=1))
    escala = torso[1:]
    with np.errstate(invalid="ignore"):
        es_cruce = (cruzado < ratio * directo) & (directo / escala > min_mov)
    cruces[1:] = np.nan_to_num(es_cruce, nan=0).astype(bool)
    return cruces


def detectar_saltos_persona(xy, torso):
    """Frames en los que la cadera se teletransporta o el torso cambia de escala."""
    cadera = punto_medio(xy["left_hip"], xy["right_hip"])
    n = len(cadera)
    saltos = np.zeros(n, dtype=bool)
    if n < 2:
        return saltos
    mov = np.linalg.norm(cadera[1:] - cadera[:-1], axis=1) / torso[:-1]
    cambio = np.abs(torso[1:] - torso[:-1]) / torso[:-1]
    with np.errstate(invalid="ignore"):
        s = (mov > UMBRAL_SALTO_CADERA) | (cambio > UMBRAL_CAMBIO_ESCALA)
    saltos[1:] = np.nan_to_num(s, nan=0).astype(bool)
    return saltos


def evaluar(ruta_csv, ancho, alto, lado="right"):
    n, xy, vis = cargar(ruta_csv, ancho, alto)
    torso = longitud_torso(xy)
    detectado = ~np.isnan(torso)
    n_det = int(detectado.sum())

    marcas = pd.DataFrame({"frame": np.arange(n), "detectado": detectado})

    # picos en las articulaciones del brazo que golpea y en ambas piernas
    distales = [f"{lado}_elbow", f"{lado}_wrist", "left_knee", "right_knee",
                "left_ankle", "right_ankle"]
    pico_any = np.zeros(n, dtype=bool)
    for nombre in distales:
        p = detectar_picos(xy[nombre], torso)
        marcas[f"pico_{nombre}"] = p
        pico_any |= p

    cruce_any = np.zeros(n, dtype=bool)
    for parte in PARES_LR:
        c = detectar_cruces(xy[f"left_{parte}"], xy[f"right_{parte}"], torso)
        marcas[f"cruce_{parte}"] = c
        cruce_any |= c

    saltos = detectar_saltos_persona(xy, torso)
    marcas["salto_persona"] = saltos

    vis_muneca = vis[f"{lado}_wrist"]
    vis_codo = vis[f"{lado}_elbow"]
    muneca_baja = ~(np.nan_to_num(vis_muneca, nan=0) >= UMBRAL_VISIBILITY)
    marcas["muneca_baja"] = muneca_baja & detectado

    sospechoso = pico_any | cruce_any | saltos
    marcas["sospechoso"] = sospechoso

    # estabilidad de huesos: dispersion robusta de la longitud relativa al torso
    huesos = {}
    for nombre, (a, b) in HUESOS.items():
        lados = [lado] if nombre in ("brazo", "antebrazo") else ["left", "right"]
        vals = []
        for s in lados:
            l = np.linalg.norm(xy[f"{s}_{a}"] - xy[f"{s}_{b}"], axis=1) / torso
            vals.append(l[~np.isnan(l)])
        v = np.concatenate(vals) if vals else np.array([])
        if len(v):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            huesos[nombre] = (q3 - q1) / med if med else np.nan
        else:
            huesos[nombre] = np.nan

    def pct(mask):
        return 100 * mask.sum() / n_det if n_det else 0.0

    resumen = {
        "frames": n,
        "pct_detectado": 100 * n_det / n if n else 0.0,
        "torso_px_mediana": float(np.nanmedian(torso)) if n_det else np.nan,
        "vis_muneca_media": float(np.nanmean(vis_muneca)) if n_det else np.nan,
        "pct_muneca_ok": 100 - pct(marcas["muneca_baja"].to_numpy()),
        "vis_codo_media": float(np.nanmean(vis_codo)) if n_det else np.nan,
        "pct_picos": pct(pico_any),
        "pct_cruces": pct(cruce_any),
        "n_saltos_persona": int(saltos.sum()),
        "pct_sospechoso": pct(sospechoso),
    }
    for nombre, v in huesos.items():
        resumen[f"iqr_{nombre}"] = v
    return resumen, marcas


def dimensiones(ruta_video):
    c = cv2.VideoCapture(ruta_video)
    if not c.isOpened():
        sys.exit(f"ERROR: OpenCV no puede abrir {ruta_video}")
    ancho, alto = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return ancho, alto


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pares", nargs="+",
                        help="Pares video.mp4 landmarks.csv (el video da la resolucion)")
    parser.add_argument("--lado", default="right", choices=["right", "left"],
                        help="Brazo que golpea")
    parser.add_argument("--marcas-dir", help="Si se indica, guarda las marcas por frame")
    parser.add_argument("--out", help="CSV con el resumen de todos los clips")
    args = parser.parse_args()

    if len(args.pares) % 2:
        sys.exit("ERROR: los argumentos van por pares video csv")

    filas = []
    for ruta_video, ruta_csv in zip(args.pares[::2], args.pares[1::2]):
        ancho, alto = dimensiones(ruta_video)
        resumen, marcas = evaluar(ruta_csv, ancho, alto, args.lado)
        nombre = os.path.basename(ruta_csv).replace("_landmarks.csv", "")
        filas.append({"clip": nombre, **resumen})
        if args.marcas_dir:
            os.makedirs(args.marcas_dir, exist_ok=True)
            marcas.to_csv(os.path.join(args.marcas_dir, f"{nombre}_marcas.csv"), index=False)

    tabla = pd.DataFrame(filas).set_index("clip")
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", "{:.2f}".format):
        print(tabla)
    if args.out:
        tabla.to_csv(args.out)


if __name__ == "__main__":
    main()
