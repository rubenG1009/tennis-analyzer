#!/usr/bin/env python3
"""Descarga los clips de referencia de references/manifest.csv a data/raw/ref/.

Solo video (sin audio: no hace falta y asi no hay que mezclar pistas), H.264
para que OpenCV lo lea sin sorpresas, maximo 1080p. Si el manifiesto trae
inicio_s/fin_s se descarga solo ese tramo.

yt-dlp se ejecuta con `uvx` y no desde el venv: las versiones de yt-dlp que
siguen funcionando con YouTube exigen Python >= 3.10 y el proyecto esta en 3.9
(la ultima compatible, 2025.10.14, ya falla con "The page needs to be
reloaded"). Tambien necesita un runtime de JS (node). ffmpeg sale de
imageio-ffmpeg (pip install imageio-ffmpeg), un binario estatico que evita
instalar nada en el sistema. Idempotente: si el fichero ya existe no lo vuelve
a bajar.
"""

import argparse
import csv
import os
import subprocess
import sys

import imageio_ffmpeg

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATO = "bv*[vcodec^=avc1][height<=1080]/bv*[height<=1080]/b"


def descargar(fila, out_dir):
    destino = os.path.join(out_dir, f"{fila['nombre']}.mp4")
    if os.path.isfile(destino):
        print(f"ya existe, salto: {destino}")
        return True
    cmd = [
        "uvx", "--python", "3.12", "--from", "yt-dlp[default]", "yt-dlp",
        "--no-warnings", "--no-playlist", "--js-runtimes", "node",
        "-f", FORMATO, "--remux-video", "mp4",
        "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
        "-o", destino,
    ]
    if fila["inicio_s"] or fila["fin_s"]:
        cmd += ["--download-sections", f"*{fila['inicio_s'] or 0}-{fila['fin_s'] or 'inf'}",
                "--force-keyframes-at-cuts"]
    cmd.append(f"https://www.youtube.com/watch?v={fila['youtube_id']}")
    print(f"descargando {fila['nombre']} ...")
    return subprocess.run(cmd).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.path.join(RAIZ, "references", "manifest.csv"))
    parser.add_argument("--out-dir", default=os.path.join(RAIZ, "data", "raw", "ref"))
    parser.add_argument("--solo", nargs="*", help="Nombres concretos a descargar")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest, newline="") as f:
        filas = list(csv.DictReader(f))
    if args.solo:
        filas = [f for f in filas if f["nombre"] in args.solo]

    fallidos = [f["nombre"] for f in filas if not descargar(f, args.out_dir)]
    if fallidos:
        sys.exit(f"Fallaron: {', '.join(fallidos)}")


if __name__ == "__main__":
    main()
