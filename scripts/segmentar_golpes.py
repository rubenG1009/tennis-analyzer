#!/usr/bin/env python3
"""Corta un CSV de landmarks en golpes individuales.

Hay dos detectores, porque el saque y los golpes de fondo no se parecen:

Saque. El impacto es geometrico: la muneca dominante en su punto mas alto,
por encima de la cabeza (mas de ALTURA_SAQUE torsos sobre la nariz). Se toma
un impacto por cada episodio con la muneca arriba (episodios separados por
menos de UNION_EPISODIOS frames se unen). Sobre los 5 saques anotados
de references/golpes.csv acierta 5 de 5 sin falsos positivos, con el codo a
166-177 grados en el impacto. La velocidad no sirve aqui: un saque tiene picos
en el lanzamiento, el trofeo, la caida de la raqueta y la terminacion.

Golpes de fondo. Senal: velocidad de la muneca dominante RELATIVA A LA CADERA y en torsos por
frame. Restar la cadera quita el desplazamiento del jugador y los barridos de
camara; dividir por el torso quita la distancia a camara.

Los umbrales son relativos al propio tramo y no absolutos, porque en los clips
de camara lenta no se conoce el factor de ralentizacion: 0.02 torsos/frame es
un golpe a 1000 fps reproducido a 60 o un jugador quieto en tiempo real. Un
golpe es un pico de velocidad que destaca sobre el ruido del tramo
(PROMINENCIA_RELATIVA veces su mediana).

El impacto se estima en el pico de velocidad de la muneca. Funciona en video
a velocidad real, que es lo que se graba con un movil. NO funciona en camara
lenta editada con rampas (la reproduccion acelera tras el impacto y el pico
cae en la terminacion: Djokovic a 1000 fps, 280 frames tarde). Los picos de
fondo se descartan mientras la otra mano esta por encima de la cabeza (el
lanzamiento de bola, que solo ocurre en el saque) y hasta EXCLUSION_SAQUE_S
despues; una ventana fija en segundos no basta en camara lenta.

Tipo de golpe: "saque" o "fondo". Derecha frente a reves no se clasifica: la
distancia entre munecas en el impacto no separa el reves a dos manos (derechas
anotadas con 0.02-0.42 torsos, el mismo rango que el reves), y el lado del
cuerpo en que ocurre el golpe es profundidad, que no se usa. El tipo de golpe
lo indica el usuario.

Un pico solo cuenta como golpe si hay raqueta a menos de DIST_RAQUETA torsos de
la muneca dominante en alguno de los frames cercanos (requiere
<nombre>_raqueta.csv). Asi se descartan gestos sin raqueta y personas que no
son el jugador (en el clip de Ruud el modelo sigue 100 frames a un entrenador).
El filtro solo se aplica si en el clip la raqueta aparece junto a una mano en
al menos COBERTURA_MIN_RAQUETA de los frames: con camara cercana (raqueta fuera
de plano) o jugador lejos el detector la pierde y el filtro eliminaria golpes
reales (Sinner: 42 % de cobertura y 0 golpes encontrados con el filtro).

Ningun golpe cruza un corte de escena ni un tramo sin deteccion.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corregir_esqueleto import cargar_cajas, distancia_a_caja  # noqa: E402
from evaluar_esqueleto import (  # noqa: E402
    cargar, detectar_saltos_persona, dimensiones, longitud_torso, punto_medio,
)

SIGMA_SUAVIZADO = 2.0        # frames
# Calibrado con los 33 golpes anotados a velocidad real (references/golpes.csv):
# con 6 solo se encontraba el 27 %, con 3 el 85-94 % con precision ~80-85 %.
PROMINENCIA_RELATIVA = 3.0   # veces la mediana de la velocidad del tramo
SEPARACION_MIN_S = 0.6       # segundos de video entre impactos
MARGEN_VENTANA_S = 0.6       # segundos antes y despues del impacto
DIST_RAQUETA = 0.6           # torsos
VENTANA_RAQUETA = 5          # frames alrededor del pico donde buscar la raqueta
COBERTURA_MIN_RAQUETA = 0.5  # fraccion de frames con raqueta junto a una mano
MIN_TRAMO = 15               # frames; tramos mas cortos no se analizan
ALTURA_SAQUE = 0.8           # torsos de la muneca por encima de la nariz
EXCLUSION_SAQUE_S = 1.0      # segundos tras el lanzamiento sin golpes de fondo
ALTURA_LANZAMIENTO = 0.3     # torsos de la otra muneca sobre la nariz
UNION_EPISODIOS = 10         # frames


def velocidad_relativa(muneca, cadera, torso):
    """Velocidad de la muneca respecto a la cadera, en torsos por frame.

    Funcion pura. Devuelve (n,) con NaN donde falta algun dato.
    """
    torso_s = pd.Series(torso).rolling(15, center=True, min_periods=1).median().to_numpy()
    rel = (muneca - cadera) / torso_s[:, None]
    v = np.full(len(rel), np.nan)
    v[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    return v


def tramos(validos):
    """Lista de (inicio, fin) de los tramos contiguos de True. Funcion pura."""
    bordes = np.diff(np.r_[0, validos.astype(int), 0])
    return list(zip(np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1)))


def picos_de_golpe(v, validos, separacion_min, prominencia_rel=None):
    """Frames de pico de velocidad que destacan sobre su tramo. Funcion pura."""
    if prominencia_rel is None:
        prominencia_rel = PROMINENCIA_RELATIVA
    picos = []
    for a, b in tramos(validos):
        if b - a < MIN_TRAMO:
            continue
        s = gaussian_filter1d(np.nan_to_num(v[a:b]), SIGMA_SUAVIZADO)
        base = np.median(s)
        if base <= 0:
            continue
        p, _ = find_peaks(s, prominence=prominencia_rel * base, distance=separacion_min)
        picos.extend(int(a + i) for i in p)
    return picos


def impactos_de_saque(altura, validos):
    """Un impacto por episodio con la muneca por encima de ALTURA_SAQUE.

    altura: (n,) torsos de la muneca sobre la nariz (positivo = mas alta).
    Funcion pura. Devuelve los frames del maximo de cada episodio.
    """
    s = pd.Series(np.where(validos, altura, np.nan)).rolling(5, center=True, min_periods=1).mean()
    episodios = tramos((s > ALTURA_SAQUE).to_numpy())
    unidos = []
    for a, b in episodios:
        if unidos and a - unidos[-1][1] < UNION_EPISODIOS:
            unidos[-1] = (unidos[-1][0], b)
        else:
            unidos.append((a, b))
    # el maximo solo entre frames validos: la media movil rellena los huecos
    # y el maximo podia caer en un frame sin deteccion
    s = np.where(validos, s.to_numpy(), np.nan)
    return [int(a + np.nanargmax(s[a:b])) for a, b in unidos if np.isfinite(s[a:b]).any()]


def zona_de_lanzamiento(altura_otra, margen):
    """Frames con la otra mano por encima de la cabeza, dilatados `margen`
    frames hacia delante y la mitad hacia atras (el gesto empieza antes de que
    la mano supere la cabeza). Funcion pura."""
    arriba = np.nan_to_num(altura_otra, nan=-np.inf) > ALTURA_LANZAMIENTO
    zona = arriba.copy()
    for a, b in tramos(arriba):
        zona[max(a - margen // 2, 0):min(b + margen, len(zona))] = True
    return zona


def cobertura_raqueta(cajas, muneca, otra, torso):
    """Fraccion de frames con una raqueta a menos de 1 torso de alguna muneca."""
    n = len(cajas)
    cerca = 0
    for t in range(n):
        if np.isnan(torso[t]):
            continue
        for c in cajas[t]:
            d = min(distancia_a_caja(muneca[t], c), distancia_a_caja(otra[t], c))
            if d / torso[t] < 1.0:
                cerca += 1
                break
    return cerca / n if n else 0.0


def segmentar(ruta_csv, ancho, alto, fps, ruta_raqueta=None, ruta_cortes=None,
              lado="right", solo=None):
    """Golpes del clip. Con solo="fondo" no se buscan saques, y con
    solo="saque" no se buscan golpes de fondo. Si se sabe que tipo de golpe
    tiene el clip hay que decirlo: una preparacion alta de derecha, con la
    raqueta por encima de la cabeza, pasaba por saque y ademas excluia los
    golpes de fondo de alrededor (primer clip propio a 240 fps, 2026-10-12).
    """
    n, xy, _ = cargar(ruta_csv, ancho, alto)
    torso = longitud_torso(xy)
    cadera = punto_medio(xy["left_hip"], xy["right_hip"])
    muneca = xy[f"{lado}_wrist"]
    otra = xy[f"{'left' if lado == 'right' else 'right'}_wrist"]
    v = velocidad_relativa(muneca, cadera, torso)

    validos = ~np.isnan(v)
    validos &= ~detectar_saltos_persona(xy, torso)
    if ruta_cortes and os.path.isfile(ruta_cortes):
        for f in pd.read_csv(ruta_cortes)["frame"]:
            validos[int(f)] = False

    torso_s = pd.Series(torso).rolling(15, center=True, min_periods=1).median().to_numpy()
    altura = (xy["nose"][:, 1] - muneca[:, 1]) / torso_s
    saques = [] if solo == "fondo" else impactos_de_saque(altura, validos)

    separacion = max(1, int(SEPARACION_MIN_S * fps))
    lanzamiento = zona_de_lanzamiento((xy["nose"][:, 1] - otra[:, 1]) / torso_s,
                                      int(EXCLUSION_SAQUE_S * fps))
    exclusion = int(EXCLUSION_SAQUE_S * fps)
    if solo == "saque":
        picos = []
    else:
        picos = [p for p in picos_de_golpe(v, validos, separacion)
                 if not (solo != "fondo" and lanzamiento[p])
                 and all(abs(p - q) > exclusion for q in saques)]
    candidatos = sorted([(p, "saque") for p in saques] + [(p, "fondo") for p in picos])

    cajas = None
    if ruta_raqueta and os.path.isfile(ruta_raqueta):
        cajas = cargar_cajas(ruta_raqueta, n, ancho, alto)
        if cobertura_raqueta(cajas, muneca, otra, torso) < COBERTURA_MIN_RAQUETA:
            cajas = None

    margen = int(MARGEN_VENTANA_S * fps)
    limites = {t: (a, b) for a, b in tramos(validos) for t in range(a, b)}
    golpes = []
    for p, tipo in candidatos:
        if tipo == "fondo" and cajas is not None:
            cerca = [
                distancia_a_caja(muneca[t], c) / torso[t]
                for t in range(max(p - VENTANA_RAQUETA, 0), min(p + VENTANA_RAQUETA + 1, n))
                for c in cajas[t] if not np.isnan(torso[t])
            ]
            if not (cerca and min(cerca) <= DIST_RAQUETA):
                continue
        a, b = limites[p]
        golpes.append({
            "tipo": tipo,
            "impacto": p,
            "inicio": max(a, p - margen),
            "fin": min(b - 1, p + margen),
            "velocidad_pico": float(v[p]),
        })
    return golpes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="Video original (resolucion y fps)")
    parser.add_argument("csv", help="CSV de landmarks (mejor el _corregido)")
    parser.add_argument("--lado", default="right", choices=["right", "left"])
    parser.add_argument("--out", help="CSV de salida (por defecto <prefijo>_golpes.csv)")
    args = parser.parse_args()

    import cv2
    c = cv2.VideoCapture(args.video)
    fps = c.get(cv2.CAP_PROP_FPS)
    c.release()
    ancho, alto = dimensiones(args.video)

    prefijo = args.csv.replace("_corregido.csv", "").replace("_landmarks.csv", "")
    golpes = segmentar(args.csv, ancho, alto, fps, f"{prefijo}_raqueta.csv",
                       f"{prefijo}_cortes.csv", args.lado)
    salida = args.out or f"{prefijo}_golpes.csv"
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tipo", "inicio", "impacto", "fin", "velocidad_pico"])
        w.writeheader()
        w.writerows(golpes)
    print(f"{os.path.basename(salida)}: {len(golpes)} golpes "
          f"{[(g['tipo'], g['impacto']) for g in golpes]}")


if __name__ == "__main__":
    main()
