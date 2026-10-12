#!/usr/bin/env python3
"""Corrige inversiones izquierda/derecha y picos aislados de un CSV de landmarks.

Escribe un CSV nuevo (<nombre>_corregido.csv) con el mismo formato largo mas una
columna `correccion`; el CSV original no se toca.

Inversiones. BlazePose no distingue bien si el jugador mira a camara o esta de
espaldas: cuando duda, invierte el cuerpo entero (hombros, caderas, brazos y
piernas cambian de lado a la vez) y la visibility sigue alta. Como la
inversion es de cuerpo entero, se decide por frame entre dos estados, "tal
cual" o "volteado", con programacion dinamica (Viterbi): el coste de pasar de
un frame al siguiente es lo que se mueven las articulaciones, y voltear un
frame tiene una penalizacion fija pequena. El resultado es la asignacion mas
continua que se aparta lo menos posible de lo que dijo el modelo.

Ancla de raqueta. La continuidad sola no sabe cual de las dos orientaciones es
la buena: si el modelo se equivoca durante la mayor parte de un tramo, el
resultado es coherente pero coherentemente al reves (pasa en el saque de
Roddick de los clips de referencia). Si hay <nombre>_raqueta.csv (extract_pose
--raqueta), en los frames donde la raqueta esta claramente mas cerca de una
muneca que de la otra, el estado que deja la mano dominante (--lado) lejos de
la raqueta paga PESO_RAQUETA. Con las dos manos en la raqueta (reves a dos
manos, preparacion) no hay evidencia y no se aplica nada.

Picos. Un landmark que evaluar_esqueleto.detectar_picos marca como pico se
sustituye por el punto medio de sus dos vecinos.

Ninguna correccion cruza un frame sin deteccion ni un salto de persona: ahi la
cadena se corta y cada tramo se resuelve por separado.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_esqueleto import (  # noqa: E402
    cargar, detectar_picos, detectar_saltos_persona, dimensiones, longitud_torso,
)
from extract_pose import NOMBRES_LANDMARKS  # noqa: E402

# Partes con pareja izquierda/derecha que cuentan para decidir el volteo. La
# cara se excluye: sus landmarks son muy ruidosos de perfil y de espaldas.
PARTES_VOLTEO = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
PENALIZACION_VOLTEO = 0.1   # torsos por frame volteado
PESO_RAQUETA = 1.0          # torsos por frame con la mano dominante lejos de la raqueta
# La raqueta solo cuenta cuando una muneca esta practicamente en ella y la otra
# claramente lejos: agarre a una mano. Con margenes mas laxos (1.0 / 0.3) la
# posicion de espera (las dos manos en la raqueta) y la terminacion (la raqueta
# sobre el hombro contrario, recogida por la otra mano) dan evidencia falsa;
# se vio en la derecha de Dimitrov de los clips de referencia.
DIST_MAX_RAQUETA = 0.3      # torsos; la muneca mas cercana tiene que estar a esto
MARGEN_RAQUETA = 0.8        # torsos de diferencia minima entre las dos munecas


def nombre_espejo(nombre):
    if nombre.startswith("left_"):
        return "right_" + nombre[5:]
    if nombre.startswith("right_"):
        return "left_" + nombre[6:]
    return nombre


def coste_transicion(izq_prev, der_prev, izq, der, torso):
    """Coste (2x2) de pasar del estado s_prev al estado s entre dos frames.

    izq_*/der_* son arrays (k, 2) con las k partes de PARTES_VOLTEO. Funcion pura.
    """
    prev = {0: (izq_prev, der_prev), 1: (der_prev, izq_prev)}
    cur = {0: (izq, der), 1: (der, izq)}
    c = np.zeros((2, 2))
    for sp in (0, 1):
        for s in (0, 1):
            d = (np.linalg.norm(cur[s][0] - prev[sp][0], axis=1).sum()
                 + np.linalg.norm(cur[s][1] - prev[sp][1], axis=1).sum())
            c[sp, s] = d / torso
    return c


def distancia_a_caja(punto, caja):
    """Distancia de un punto (x, y) a un rectangulo (x0, y0, x1, y1); 0 si esta dentro."""
    dx = max(caja[0] - punto[0], 0, punto[0] - caja[2])
    dy = max(caja[1] - punto[1], 0, punto[1] - caja[3])
    return float(np.hypot(dx, dy))


def costes_raqueta(cajas_por_frame, muneca_dom, muneca_otra, torso):
    """Coste de emision (n, 2) segun la raqueta. Funcion pura.

    cajas_por_frame: lista de n listas de cajas (x0, y0, x1, y1) en pixeles.
    muneca_dom/muneca_otra: (n, 2), munecas tal como las etiqueto el modelo.
    Columna 0 = no voltear, 1 = voltear.
    """
    n = len(cajas_por_frame)
    costes = np.zeros((n, 2))
    for t, cajas in enumerate(cajas_por_frame):
        if not cajas or np.isnan(torso[t]) or np.isnan(muneca_dom[t]).any():
            continue
        # la caja que este mas cerca de cualquiera de las dos munecas
        d = [(distancia_a_caja(muneca_dom[t], c), distancia_a_caja(muneca_otra[t], c))
             for c in cajas]
        d_dom, d_otra = min(d, key=min)
        d_dom, d_otra = d_dom / torso[t], d_otra / torso[t]
        if min(d_dom, d_otra) > DIST_MAX_RAQUETA or abs(d_dom - d_otra) < MARGEN_RAQUETA:
            continue
        if d_dom < d_otra:
            costes[t, 1] = PESO_RAQUETA   # voltear alejaria la mano dominante
        else:
            costes[t, 0] = PESO_RAQUETA   # el modelo tiene las manos cambiadas
    return costes


def decidir_volteos(izq, der, torso, cortes, penalizacion=PENALIZACION_VOLTEO,
                    emision_extra=None):
    """Viterbi de dos estados. izq/der: (n, k, 2). cortes: bool (n,) donde la
    cadena se reinicia (frame sin deteccion o salto de persona). emision_extra:
    (n, 2) opcional, coste adicional por frame y estado (la raqueta).

    Devuelve un array booleano (n,): True si el frame debe voltearse. Funcion pura.
    """
    n = len(izq)
    volteo = np.zeros(n, dtype=bool)
    if emision_extra is None:
        emision_extra = np.zeros((n, 2))
    base = np.array([0.0, penalizacion])
    inicio = 0
    while inicio < n:
        # tramo continuo [inicio, fin)
        fin = inicio + 1
        while fin < n and not cortes[fin]:
            fin += 1
        if not np.isnan(torso[inicio]):
            acum = base + emision_extra[inicio]
            atras = np.zeros((fin - inicio, 2), dtype=int)
            for t in range(inicio + 1, fin):
                trans = coste_transicion(izq[t - 1], der[t - 1], izq[t], der[t], torso[t])
                total = acum[:, None] + trans
                atras[t - inicio] = np.argmin(total, axis=0)
                acum = total[atras[t - inicio], [0, 1]] + base + emision_extra[t]
            s = int(np.argmin(acum))
            for t in range(fin - 1, inicio - 1, -1):
                volteo[t] = bool(s)
                s = atras[t - inicio, s]
        inicio = fin
    return volteo


def cargar_cajas(ruta_raqueta, n, ancho, alto):
    cajas = [[] for _ in range(n)]
    if ruta_raqueta and os.path.isfile(ruta_raqueta):
        for fila in pd.read_csv(ruta_raqueta).itertuples(index=False):
            if fila.frame < n:
                cajas[fila.frame].append((fila.x0 * ancho, fila.y0 * alto,
                                          fila.x1 * ancho, fila.y1 * alto))
    return cajas


def corregir(ruta_csv, ancho, alto, ruta_salida, ruta_raqueta=None, ruta_cortes=None,
             lado="right"):
    n, xy, vis = cargar(ruta_csv, ancho, alto)
    torso = longitud_torso(xy)
    sin_det = np.isnan(torso)
    cortes = sin_det | detectar_saltos_persona(xy, torso)
    if ruta_cortes and os.path.isfile(ruta_cortes):
        for f in pd.read_csv(ruta_cortes)["frame"]:
            cortes[int(f)] = True

    otro = "left" if lado == "right" else "right"
    cajas = cargar_cajas(ruta_raqueta, n, ancho, alto)
    emision = costes_raqueta(cajas, xy[f"{lado}_wrist"], xy[f"{otro}_wrist"], torso)

    izq = np.stack([xy[f"left_{p}"] for p in PARTES_VOLTEO], axis=1)
    der = np.stack([xy[f"right_{p}"] for p in PARTES_VOLTEO], axis=1)
    volteo = decidir_volteos(izq, der, torso, cortes, emision_extra=emision)

    # aplicar volteo a todos los landmarks con pareja (cara incluida)
    xy_c = {k: v.copy() for k, v in xy.items()}
    vis_c = {k: v.copy() for k, v in vis.items()}
    for nombre in NOMBRES_LANDMARKS:
        espejo = nombre_espejo(nombre)
        if espejo != nombre:
            xy_c[nombre][volteo] = xy[espejo][volteo]
            vis_c[nombre][volteo] = vis[espejo][volteo]

    # picos: sustituir por el punto medio de los vecinos, sin cruzar cortes
    picos = {}
    for nombre in NOMBRES_LANDMARKS:
        p = detectar_picos(xy_c[nombre], torso)
        p &= ~cortes & ~np.roll(cortes, -1)   # ni el frame del corte ni el anterior
        p[[0, -1]] = False
        idx = np.flatnonzero(p)
        xy_c[nombre][idx] = (xy_c[nombre][idx - 1] + xy_c[nombre][idx + 1]) / 2
        picos[nombre] = p

    # reescribir en formato largo conservando t_ms y z del original
    df = pd.read_csv(ruta_csv)
    t_ms = df.groupby("frame")["t_ms"].first().reindex(range(n)).to_numpy()
    z = {nombre: g.set_index("frame")["z"].reindex(range(n)).to_numpy()
         for nombre, g in df.groupby("landmark")}
    with open(ruta_salida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(df.columns) + ["correccion"])
        for frame in range(n):
            for i, nombre in enumerate(NOMBRES_LANDMARKS):
                if sin_det[frame]:
                    w.writerow([frame, t_ms[frame], i, nombre, "", "", "", "", ""])
                    continue
                espejo = nombre_espejo(nombre) if volteo[frame] else nombre
                x, y = xy_c[nombre][frame]
                etiquetas = []
                if espejo != nombre:
                    etiquetas.append("volteo")
                if picos[nombre][frame]:
                    etiquetas.append("pico")
                w.writerow([frame, t_ms[frame], i, nombre,
                            f"{x / ancho:.6f}", f"{y / alto:.6f}", f"{z[espejo][frame]:.6f}",
                            f"{vis_c[nombre][frame]:.6f}", "+".join(etiquetas)])
    return {
        "frames": n,
        "frames_volteados": int(volteo.sum()),
        "landmarks_pico": int(sum(p.sum() for p in picos.values())),
        "frames_con_evidencia_raqueta": int((emision.sum(axis=1) > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="Video original (para la resolucion)")
    parser.add_argument("csv", help="CSV de landmarks de extract_pose.py")
    parser.add_argument("--out", help="CSV de salida (por defecto <csv>_corregido.csv)")
    parser.add_argument("--lado", default="right", choices=["right", "left"],
                        help="Mano dominante del jugador")
    args = parser.parse_args()

    ancho, alto = dimensiones(args.video)
    salida = args.out or args.csv.replace("_landmarks.csv", "_corregido.csv")
    prefijo = args.csv.replace("_landmarks.csv", "")
    res = corregir(args.csv, ancho, alto, salida, f"{prefijo}_raqueta.csv",
                   f"{prefijo}_cortes.csv", args.lado)
    print(f"{os.path.basename(salida)}: {res['frames_volteados']} frames volteados "
          f"de {res['frames']}, {res['landmarks_pico']} landmarks con pico, "
          f"{res['frames_con_evidencia_raqueta']} frames con evidencia de raqueta")


if __name__ == "__main__":
    main()
