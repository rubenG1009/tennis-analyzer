#!/usr/bin/env python3
"""Metricas biomecanicas 2D por golpe.

Todas en el plano de la imagen (z no se usa, ver docs/decisiones.md), en
grados o en longitudes de torso para que sean comparables entre clips de
distinta resolucion y distancia a camara. Cada valor se toma como mediana de
los frames [impacto - SUAVIZADO, impacto + SUAVIZADO] para no depender de un
frame suelto.

Golpes de fondo y saque:
  codo_impacto       angulo hombro-codo-muneca del brazo dominante (180 = recto)
  rodilla_min        angulo cadera-rodilla-tobillo minimo (la pierna mas
                     flexionada) en la ventana de carga antes del impacto. La
                     ventana se mide en frames del propio gesto y no en
                     segundos, porque en camara lenta no se conoce la escala
                     de tiempo: empieza 2 veces mas atras que el ultimo frame
                     en que la muneca iba a menos del 20 % de su velocidad en
                     el impacto (en el saque: desde el lanzamiento de bola)
  tronco_impacto     inclinacion del tronco respecto a la vertical (0 = erguido)

Solo fondo:
  altura_impacto     altura de la muneca: 0 = cadera, 1 = hombros
  ancho_apoyos       distancia entre tobillos en el impacto, en torsos
  impacto_delante    cuanto esta la muneca por delante de la cadera, en torsos
                     (negativo = tarde). "Delante" es hacia donde mira la
                     cabeza (nariz respecto al punto medio de las orejas): en
                     el impacto el jugador mira la bola. Solo vista lateral.

Solo saque:
  alcance            muneca por encima de la nariz, en torsos

Limitaciones de una sola camara: los angulos se acortan cuando el segmento
apunta hacia la camara. codo y rodilla son fiables en vista lateral; en vista
frontal o de espaldas una rodilla flexionada hacia la camara parece recta. Por
eso cada golpe lleva su vista y las referencias se agregan por vista.

Calidad: si el hombro, el codo o la muneca dominantes no alcanzan UMBRAL_VIS en
el impacto, las metricas del brazo (codo, altura, delante, alcance) quedan
vacias. Un numero sobre un brazo que el modelo no ve es ruido con aspecto de
dato (en Sinner, frame 1017: codo a 23 grados con visibility 0.14).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_esqueleto import cargar, longitud_torso, punto_medio  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUAVIZADO = 2              # frames a cada lado del impacto
FRACCION_INICIO_GESTO = 0.2  # el gesto empieza cuando la muneca va a menos de esto
MAX_VENTANA_CARGA = 400    # frames; tope por si la muneca nunca se para
UMBRAL_VIS = 0.5
METRICAS_FONDO = ["codo_impacto", "rodilla_min", "tronco_impacto", "altura_impacto",
                  "ancho_apoyos", "impacto_delante"]
METRICAS_SAQUE = ["codo_impacto", "rodilla_min", "tronco_impacto", "alcance"]


def angulo(a, b, c):
    """Angulo en b (grados) de los puntos a-b-c; arrays (n, 2). Funcion pura."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def inclinacion(abajo, arriba):
    """Angulo con la vertical del segmento abajo->arriba (grados). Funcion pura.

    En imagen la y crece hacia abajo; 0 = vertical, 90 = horizontal.
    """
    d = arriba - abajo
    return np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))


def series(xy, lado):
    """Series por frame de todas las magnitudes que usan las metricas."""
    otro = "left" if lado == "right" else "right"
    torso = pd.Series(longitud_torso(xy)).rolling(15, center=True, min_periods=1).median().to_numpy()
    cadera = punto_medio(xy["left_hip"], xy["right_hip"])
    hombros = punto_medio(xy["left_shoulder"], xy["right_shoulder"])
    muneca = xy[f"{lado}_wrist"]
    rodillas = np.column_stack([
        angulo(xy[f"{s}_hip"], xy[f"{s}_knee"], xy[f"{s}_ankle"]) for s in (lado, otro)])
    return {
        "torso": torso,
        "codo": angulo(xy[f"{lado}_shoulder"], xy[f"{lado}_elbow"], muneca),
        "rodilla": rodillas.min(axis=1),
        "tronco": inclinacion(cadera, hombros),
        # proyeccion vertical de la muneca entre cadera (0) y hombros (1)
        "altura": (cadera[:, 1] - muneca[:, 1]) / (cadera[:, 1] - hombros[:, 1]),
        "apoyos": np.linalg.norm(xy["left_ankle"] - xy["right_ankle"], axis=1) / torso,
        "alcance": (xy["nose"][:, 1] - muneca[:, 1]) / torso,
        "alcance_otra": (xy["nose"][:, 1] - xy[f"{otro}_wrist"][:, 1]) / torso,
        "mirada": np.sign(xy["nose"][:, 0] - punto_medio(xy["left_ear"], xy["right_ear"])[:, 0]),
        "velocidad": np.r_[np.nan, np.linalg.norm(np.diff(
            (muneca - cadera) / torso[:, None], axis=0), axis=1)],
        "muneca": muneca,
        "cadera": cadera,
    }


def mediana_en(serie, t, k=SUAVIZADO):
    a, b = max(t - k, 0), min(t + k + 1, len(serie))
    v = serie[a:b]
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def impacto_delante(s, t, k=SUAVIZADO):
    """Distancia horizontal muneca-cadera en la direccion en que mira la cabeza."""
    a, b = max(t - k, 0), min(t + k + 1, len(s["muneca"]))
    mirada = np.nanmedian(s["mirada"][a:b])
    if not mirada or np.isnan(mirada):
        return np.nan
    rel = (s["muneca"][a:b, 0] - s["cadera"][a:b, 0]) * mirada / s["torso"][a:b]
    return float(np.nanmedian(rel))


def inicio_ventana_carga(s, t, tipo):
    """Primer frame de la ventana donde se busca la flexion de rodillas.

    Saque: desde que la otra mano sube por encima de la cabeza (lanzamiento),
    o 2 duraciones de gesto si no se ve. Fondo: 2 veces la duracion del gesto
    hacia atras, siendo el gesto el tramo final con la muneca por encima de
    FRACCION_INICIO_GESTO de su velocidad en el impacto. Funcion pura.
    """
    v = pd.Series(s["velocidad"]).rolling(5, center=True, min_periods=1).mean().to_numpy()
    if tipo == "saque":
        arriba = np.flatnonzero(np.nan_to_num(s["alcance_otra"][max(t - MAX_VENTANA_CARGA, 0):t],
                                              nan=-1) > 0.3)
        if len(arriba):
            return max(t - MAX_VENTANA_CARGA, 0) + int(arriba[0])
    ref = np.nanmax(v[max(t - SUAVIZADO, 0):t + SUAVIZADO + 1]) if t < len(v) else np.nan
    t0 = t
    while t0 > 0 and t - t0 < MAX_VENTANA_CARGA and not (v[t0] < FRACCION_INICIO_GESTO * ref):
        t0 -= 1
    return max(t - 2 * (t - t0), t - MAX_VENTANA_CARGA, 0)


def metricas(s, t, tipo, fps, vis_brazo):
    rodilla = s["rodilla"][inicio_ventana_carga(s, t, tipo):t + 1]
    m = {
        "codo_impacto": mediana_en(s["codo"], t),
        "rodilla_min": float(np.nanmin(rodilla)) if np.isfinite(rodilla).any() else np.nan,
        "tronco_impacto": mediana_en(s["tronco"], t),
        "vis_brazo": vis_brazo,
    }
    if tipo == "saque":
        m["alcance"] = mediana_en(s["alcance"], t)
    else:
        m["altura_impacto"] = mediana_en(s["altura"], t)
        m["ancho_apoyos"] = mediana_en(s["apoyos"], t)
        m["impacto_delante"] = impacto_delante(s, t)
    if not vis_brazo >= UMBRAL_VIS:
        for clave in ("codo_impacto", "altura_impacto", "impacto_delante", "alcance"):
            if clave in m:
                m[clave] = np.nan
    return m


def medir_golpes(ruta_csv, ancho, alto, fps, golpes, lado="right", desplazamiento=0):
    """golpes: iterable de (impacto, tipo). Devuelve lista de dicts de metricas."""
    n, xy, vis = cargar(ruta_csv, ancho, alto)
    s = series(xy, lado)
    filas = []
    for impacto, tipo in golpes:
        t = int(np.clip(impacto + desplazamiento, 0, n - 1))
        v = np.nanmin([vis[f"{lado}_{p}"][t] for p in ("shoulder", "elbow", "wrist")])
        filas.append({"impacto": impacto, **metricas(s, t, "saque" if tipo == "saque" else "fondo",
                                                    fps, float(v))})
    return filas


def info_video(ruta):
    c = cv2.VideoCapture(ruta)
    if not c.isOpened():
        sys.exit(f"ERROR: OpenCV no puede abrir {ruta}")
    r = c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return r


def referencias(dir_procesado, dir_videos, anotaciones, sensibilidad):
    anot = pd.concat([pd.read_csv(a) for a in anotaciones], ignore_index=True)
    filas = []
    for clip, g in anot.groupby("clip"):
        fps, ancho, alto = info_video(os.path.join(dir_videos, f"{clip}.mp4"))
        ruta = os.path.join(dir_procesado, f"{clip}_corregido.csv")
        golpes = list(zip(g["impacto"], g["golpe"]))
        base = medir_golpes(ruta, ancho, alto, fps, golpes)
        for (_, fila), m in zip(g.iterrows(), base):
            filas.append({"clip": clip, "golpe": fila["golpe"], "vista": fila["vista"], **m})
        for d in sensibilidad:
            for (_, fila), m in zip(g.iterrows(), medir_golpes(ruta, ancho, alto, fps, golpes,
                                                               desplazamiento=d)):
                filas.append({"clip": clip, "golpe": fila["golpe"], "vista": fila["vista"],
                              "desplazamiento": d, **m})
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="modo", required=True)

    p = sub.add_parser("golpes", help="Metricas de los golpes de un video")
    p.add_argument("video")
    p.add_argument("csv", help="CSV corregido")
    p.add_argument("golpes", help="CSV de segmentar_golpes.py")
    p.add_argument("--lado", default="right", choices=["right", "left"])
    p.add_argument("--out")

    r = sub.add_parser("referencias", help="Metricas de los golpes anotados de references/")
    r.add_argument("--procesado", default=os.path.join(RAIZ, "data", "processed", "ref_v2"))
    r.add_argument("--videos", default=os.path.join(RAIZ, "data", "raw", "ref"))
    r.add_argument("--anotaciones", nargs="+",
                   default=[os.path.join(RAIZ, "references", "golpes.csv"),
                            os.path.join(RAIZ, "references", "golpes_auto.csv")])
    r.add_argument("--out", default=os.path.join(RAIZ, "data", "processed", "metricas_referencia.csv"))
    args = parser.parse_args()

    if args.modo == "golpes":
        fps, ancho, alto = info_video(args.video)
        g = pd.read_csv(args.golpes)
        filas = medir_golpes(args.csv, ancho, alto, fps, zip(g["impacto"], g["tipo"]), args.lado)
        t = pd.DataFrame(filas)
        t.insert(1, "tipo", g["tipo"].to_numpy())
        salida = args.out or args.golpes.replace("_golpes.csv", "_metricas.csv")
        t.to_csv(salida, index=False, float_format="%.3f")
        print(t.round(2).to_string(index=False))
        return

    t = referencias(args.procesado, args.videos, args.anotaciones, sensibilidad=[-3, 3])
    t.to_csv(args.out, index=False, float_format="%.3f")
    base = t[t["desplazamiento"].isna()] if "desplazamiento" in t else t
    print(base.drop(columns=[c for c in ["desplazamiento"] if c in base]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
