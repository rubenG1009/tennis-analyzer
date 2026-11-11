#!/usr/bin/env python3
"""Construye references/combate_referencias.json con los clips de profesionales.

    python scripts/referencias_combate.py
    python scripts/referencias_combate.py --excluir lomachenko_3xi9 usyk_FhRN

Para cada clip de data/raw/combate/pro con esqueleto corregido en
data/processed/combate/pro, detecta y mide los golpes con un perfil de
referencia (180 cm, 75 kg) y guarda la mediana de cada metrica en unidades
independientes de la altura. Un clip solo aporta una metrica si tiene al menos
combate_referencias.MIN_GOLPES golpes fiables de ese tipo, y una metrica solo
tiene rango con MIN_CLIPS clips. Escribe tambien la tabla por clip en
references/combate_referencias_clips.csv para poder ver de quien sale cada
numero. Los clips no se redistribuyen (como en tenis): solo van los numeros.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analizar_combate as ac  # noqa: E402
import combate_perfil as cp  # noqa: E402
import combate_referencias as cr  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERFIL = cp.Perfil(altura_cm=180, peso_kg=75, experiencia_meses=36)


def medir_clip(video, procesado, nombre, perfil=PERFIL, vista="lateral"):
    import metricas_golpe
    from evaluar_esqueleto import cargar, detectar_saltos_persona, longitud_torso
    csv_corregido = os.path.join(procesado, f"{nombre}_corregido.csv")
    if not os.path.isfile(csv_corregido):
        return None
    fps, ancho, alto = metricas_golpe.info_video(video)
    n, xy, vis = cargar(csv_corregido, ancho, alto)
    return ac.analizar_xy(xy, vis, fps, perfil, vista, detectar_saltos_persona(xy, longitud_torso(xy)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", default=os.path.join(RAIZ, "data", "raw", "combate", "pro"))
    parser.add_argument("--procesado", default=os.path.join(RAIZ, "data", "processed", "combate", "pro"))
    parser.add_argument("--excluir", nargs="*", default=[], help="nombres de clip a dejar fuera")
    parser.add_argument("--salida", default=os.path.join(RAIZ, "references", "combate_referencias.json"))
    args = parser.parse_args()

    clips, filas_csv = {}, []
    for f in sorted(os.listdir(args.videos)):
        if not f.endswith(".mp4") or f[:-4] in args.excluir:
            continue
        nombre = f[:-4]
        r = medir_clip(os.path.join(args.videos, f), args.procesado, nombre)
        if r is None:
            continue
        med = cr.medianas_clip(r["golpes"], PERFIL)
        clips[nombre] = med
        s = r["sesion"]
        print(f"{nombre:18s} puños {s['puños']:3d} patadas {s['patadas']:3d} fiables {s['fiables']:3d} "
              f"aportan: {sorted(m for t in med.values() for m in t) or '-'}")
        for tipo, ms in med.items():
            for m, v in ms.items():
                filas_csv.append({"clip": nombre, "tipo": tipo, "metrica": m, "mediana_unidad_comun": round(v, 4)})
    ref = cr.construir(clips)
    cr.guardar(ref, args.salida)
    with open(args.salida.replace(".json", "_clips.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "tipo", "metrica", "mediana_unidad_comun"])
        w.writeheader()
        w.writerows(filas_csv)
    print(f"\n{sum(len(v) for v in ref.values())} metricas con rango ({cr.MIN_CLIPS}+ clips) -> {args.salida}")
    for tipo, ms in ref.items():
        for m, d in ms.items():
            print(f"  {tipo:6s} {m:24s} {d['min']:8.2f} .. {d['max']:8.2f}  ({d['n_clips']} clips)")


if __name__ == "__main__":
    main()
