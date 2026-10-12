#!/usr/bin/env python3
"""Compara los golpes detectados por segmentar_golpes con references/golpes.csv.

Dos preguntas distintas:

- Encuentra el golpe? Una deteccion empareja con un impacto anotado si cae a
  menos de VENTANA_GOLPE_S segundos de video de el. Con eso se calculan
  precision (detecciones que son golpes) y exhaustividad (golpes encontrados).
- Acierta el impacto? Para los emparejados, error en frames frente a la
  tolerancia de la anotacion.

Solo se evaluan los tramos anotados (ANOTADO_HASTA limita los clips anotados
en parte); las detecciones fuera de ellos no cuentan ni a favor ni en contra.
"""

import argparse
import os
import sys

import cv2
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENTANA_GOLPE_S = 0.5
ANOTADO_HASTA = {"fh_compilacion_espalda": 1400}


def emparejar(detectados, anotados, ventana):
    """Emparejamiento voraz por cercania. Funcion pura.

    Devuelve lista de (anotado, detectado o None) y la lista de detectados sin pareja.
    """
    libres = sorted(detectados)
    pares = []
    for a in sorted(anotados):
        candidatos = [d for d in libres if abs(d - a) <= ventana]
        if candidatos:
            d = min(candidatos, key=lambda d: abs(d - a))
            libres.remove(d)
            pares.append((a, d))
        else:
            pares.append((a, None))
    return pares, libres


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir_golpes", help="Carpeta con los <clip>_golpes.csv")
    parser.add_argument("--anotaciones", default=os.path.join(RAIZ, "references", "golpes.csv"))
    parser.add_argument("--videos", default=os.path.join(RAIZ, "data", "raw", "ref"))
    args = parser.parse_args()

    anot = pd.read_csv(args.anotaciones)
    filas = []
    falsos = []
    for clip, g in anot.groupby("clip"):
        ruta = os.path.join(args.dir_golpes, f"{clip}_golpes.csv")
        if not os.path.isfile(ruta):
            sys.exit(f"ERROR: falta {ruta}")
        c = cv2.VideoCapture(os.path.join(args.videos, f"{clip}.mp4"))
        fps = c.get(cv2.CAP_PROP_FPS)
        c.release()
        det = pd.read_csv(ruta)["impacto"].tolist()
        limite = ANOTADO_HASTA.get(clip)
        if limite is not None:
            det = [d for d in det if d <= limite]
        pares, libres = emparejar(det, g["impacto"].tolist(), int(VENTANA_GOLPE_S * fps))
        tol = dict(zip(g["impacto"], g["tolerancia"]))
        tipo = dict(zip(g["impacto"], g["golpe"]))
        for a, d in pares:
            filas.append({"clip": clip, "golpe": tipo[a], "anotado": a, "detectado": d,
                          "error": None if d is None else d - a, "tolerancia": tol[a]})
        falsos += [(clip, d) for d in libres]

    t = pd.DataFrame(filas)
    encontrados = t["detectado"].notna()
    dentro = encontrados & (t["error"].abs() <= t["tolerancia"])
    n_det = int(encontrados.sum()) + len(falsos)
    print(t.to_string(index=False))
    print()
    print(f"Golpes anotados        : {len(t)}")
    print(f"Encontrados            : {int(encontrados.sum())} "
          f"(exhaustividad {100 * encontrados.mean():.0f} %)")
    print(f"Detecciones            : {n_det}, falsas {len(falsos)} "
          f"(precision {100 * encontrados.sum() / n_det if n_det else 0:.0f} %)")
    print(f"Impacto dentro de tol. : {int(dentro.sum())} de {int(encontrados.sum())} encontrados")
    print(f"Falsos positivos       : {falsos}")
    print()
    print("Por tipo de golpe (exhaustividad):")
    print(t.assign(ok=encontrados).groupby("golpe")["ok"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
