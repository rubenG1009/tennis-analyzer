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
  terminacion        altura maxima de la muneca despues del impacto (misma
                     escala que altura_impacto: 1 = hombros). Mide si el golpe
                     termina arriba o se queda corto.
  giro_hombros       amplitud del giro de hombros en todo el gesto, en grados:
                     de la maxima torsion en la preparacion a la terminacion.
                     Ver angulo_fuera_del_plano.
  peso_delante       cadera respecto al punto medio de los pies en el impacto,
                     en torsos, hacia donde va el golpe (positivo = el peso
                     va hacia delante). Solo vista lateral.

Las tres ultimas se miden respecto al propio cuerpo y no respecto a la imagen
porque en los clips de referencia la camara sigue al jugador: cualquier
desplazamiento absoluto (cuanto avanza la cadera) mezcla jugador y camara.

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
                  "ancho_apoyos", "impacto_delante", "terminacion", "giro_hombros",
                  "peso_delante"]
PERCENTIL_ANCHO_REAL = 95  # el ancho de hombros sin escorzo, sobre todo el clip
HISTERESIS_TERMINACION = 0.15  # cuanto baja la muneca para dar el golpe por terminado
HISTERESIS_GIRO = 15.0         # grados que vuelven los hombros para dar la torsion por maxima
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


def angulo_fuera_del_plano(a, b, torso):
    """Orientacion de la linea a->b respecto al plano de la imagen, 0-180 grados.

    Con una sola camara el giro de los hombros alrededor del eje vertical solo
    se ve como escorzo: la linea entre hombros se acorta al girar hacia la
    camara. Su componente horizontal con signo, dividida por el ancho real,
    es el coseno del angulo: 0 = de espaldas a camara, 90 = de perfil,
    180 = de frente (o al reves segun el lado de la camara). El signo exige
    que izquierda y derecha esten bien, que es lo que garantiza
    corregir_esqueleto. El ancho real es el percentil PERCENTIL_ANCHO_REAL del
    clip: en cada derecha hay un momento con los hombros paralelos a la
    imagen. Funcion pura.
    """
    d = (b[:, 0] - a[:, 0]) / torso
    if not np.isfinite(d).any():
        return np.full(len(d), np.nan)
    real = np.nanpercentile(np.abs(d), PERCENTIL_ANCHO_REAL)
    return np.degrees(np.arccos(np.clip(d / real, -1, 1)))


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
        "hombros_giro": angulo_fuera_del_plano(xy["left_shoulder"], xy["right_shoulder"], torso),
        "pies": punto_medio(xy["left_ankle"], xy["right_ankle"]),
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


def direccion_golpe(s, t, k=SUAVIZADO):
    """+1 o -1 segun hacia donde va la muneca (respecto a la cadera) en el
    impacto, en x de imagen; NaN si no se ve. A diferencia de la mirada, no
    depende de la cara, que de espaldas o de perfil es ruidosa. Funcion pura."""
    rel = s["muneca"][:, 0] - s["cadera"][:, 0]
    antes, despues = rel[max(t - k - 1, 0):t], rel[t + 1:t + k + 2]
    if not (np.isfinite(antes).any() and np.isfinite(despues).any()):
        return np.nan
    return float(np.sign(np.nanmean(despues) - np.nanmean(antes))) or np.nan


def peso_delante(s, t, k=SUAVIZADO):
    """Cadera menos punto medio de los pies, en torsos, en la direccion del golpe."""
    sentido = direccion_golpe(s, t, k)
    if np.isnan(sentido):
        return np.nan
    a, b = max(t - k, 0), min(t + k + 1, len(s["torso"]))
    rel = (s["cadera"][a:b, 0] - s["pies"][a:b, 0]) * sentido / s["torso"][a:b]
    return float(np.nanmedian(rel)) if np.isfinite(rel).any() else np.nan


def hasta_el_extremo(v, t, paso, histeresis, tope=MAX_VENTANA_CARGA, oculto=None):
    """Recorre v desde t en la direccion `paso` (+1 o -1) mientras siga
    creciendo, y devuelve el indice del maximo. Para cuando v cae `histeresis`
    por debajo del maximo visto, al llegar al borde o tras `tope` frames; los
    NaN se saltan. Si `oculto` (booleano por frame) se hace True antes de
    confirmar el maximo, devuelve None: el maximo de verdad pudo estar en la
    parte que no se ve. Funcion pura.

    Sirve para encontrar la terminacion y la maxima torsion sin suponer cuanto
    dura el gesto: en camara lenta editada (con rampas de velocidad) la
    duracion que se deduce de la velocidad de la muneca no vale, y aqui no
    hace falta.
    """
    mejor, i_mejor = -np.inf, None
    i = t
    while 0 <= i < len(v) and abs(i - t) <= tope:
        if oculto is not None and oculto[i]:
            return None
        if np.isfinite(v[i]):
            if v[i] > mejor:
                mejor, i_mejor = v[i], i
            elif v[i] < mejor - histeresis:
                break
        i += paso
    return i_mejor


def suavizar(v, k=SUAVIZADO):
    return pd.Series(v).rolling(2 * k + 1, center=True, min_periods=1).median().to_numpy()


def terminacion(s, t):
    """Altura maxima de la muneca tras el impacto, o NaN si la muneca deja de
    verse antes de empezar a bajar. Visto de lado, en la terminacion la muneca
    pasa por detras de la cabeza; BlazePose la sigue colocando, mas abajo, y
    la terminacion salia cortada (primer clip propio a 240 fps: 0,79 con la
    raqueta acabando sobre el hombro)."""
    altura = suavizar(s["altura"])
    oculto = s["vis_muneca"] < UMBRAL_VIS if "vis_muneca" in s else None
    fin = hasta_el_extremo(altura, t, +1, HISTERESIS_TERMINACION, oculto=oculto)
    return np.nan if fin is None else float(altura[fin])


def giro_hombros(s, t):
    """Amplitud del giro de hombros entre la maxima torsion antes del impacto
    y el final del giro despues, en grados. Los dos extremos se buscan en la
    propia orientacion de los hombros, sin depender de la muneca."""
    giro = suavizar(s["hombros_giro"])
    if not np.isfinite(giro[t]):
        return np.nan
    lejos = np.abs(giro - giro[t])
    inicio = hasta_el_extremo(lejos, t, -1, HISTERESIS_GIRO)
    fin = hasta_el_extremo(lejos, t, +1, HISTERESIS_GIRO)
    if inicio is None or fin is None:
        return np.nan
    tramo = giro[inicio:fin + 1]
    return float(np.nanmax(tramo) - np.nanmin(tramo))


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


METRICAS_PIERNAS = ("rodilla_min", "ancho_apoyos", "peso_delante")
PUNTOS_PIERNAS = ("left_knee", "right_knee", "left_ankle", "right_ankle")


def metricas(s, t, tipo, fps, vis_brazo, vis_piernas=1.0):
    rodilla = s["rodilla"][inicio_ventana_carga(s, t, tipo):t + 1]
    m = {
        "codo_impacto": mediana_en(s["codo"], t),
        "rodilla_min": float(np.nanmin(rodilla)) if np.isfinite(rodilla).any() else np.nan,
        "tronco_impacto": mediana_en(s["tronco"], t),
        "vis_brazo": vis_brazo,
        "vis_piernas": vis_piernas,
    }
    if tipo == "saque":
        m["alcance"] = mediana_en(s["alcance"], t)
    else:
        m["altura_impacto"] = mediana_en(s["altura"], t)
        m["ancho_apoyos"] = mediana_en(s["apoyos"], t)
        m["impacto_delante"] = impacto_delante(s, t)
        m["terminacion"] = terminacion(s, t)
        m["giro_hombros"] = giro_hombros(s, t)
        m["peso_delante"] = peso_delante(s, t)
    if not vis_brazo >= UMBRAL_VIS:
        for clave in ("codo_impacto", "altura_impacto", "impacto_delante", "alcance"):
            if clave in m:
                m[clave] = np.nan
    # Igual con las piernas: si rodillas y tobillos quedan fuera del plano,
    # BlazePose se los inventa con visibility ~0.05 y salian "piernas rigidas"
    # sobre unas piernas que no estaban en el video (clip de sombras en casa,
    # 2026-10-12).
    if not vis_piernas >= UMBRAL_VIS:
        for clave in METRICAS_PIERNAS:
            if clave in m:
                m[clave] = np.nan
    return m


def medir_golpes(ruta_csv, ancho, alto, fps, golpes, lado="right", desplazamiento=0):
    """golpes: iterable de (impacto, tipo). Devuelve lista de dicts de metricas."""
    n, xy, vis = cargar(ruta_csv, ancho, alto)
    s = series(xy, lado)
    s["vis_muneca"] = np.nan_to_num(vis[f"{lado}_wrist"], nan=0.0)
    filas = []
    for impacto, tipo in golpes:
        t = int(np.clip(impacto + desplazamiento, 0, n - 1))
        v = np.nanmin([vis[f"{lado}_{p}"][t] for p in ("shoulder", "elbow", "wrist")])
        # mediana y no minimo: de perfil una pierna tapa a la otra y basta con
        # que se vean la mitad de los puntos
        vp = np.nanmedian([vis[p][t] for p in PUNTOS_PIERNAS])
        filas.append({"impacto": impacto, **metricas(s, t, "saque" if tipo == "saque" else "fondo",
                                                    fps, float(v), float(vp))})
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
