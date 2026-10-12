#!/usr/bin/env python3
"""Valida la comparacion con profesionales dejando fuera a cada jugador.

    python scripts/validar_comparacion.py

Para cada golpe y vista con referencias suficientes, analiza a cada
profesional como si fuera un usuario: su sesion es la mediana de sus golpes y
se compara con el rango de los demas (analizar_clip.comparar). Un sistema bien
calibrado deberia dejarlo "dentro del rango" casi siempre: cada vez que no, el
informe le diria a un profesional que algo merece mirarse.

Imprime el porcentaje de "dentro del rango" por metrica y en total, quien cae
fuera y, de esos, a cuantos el informe les diria "a revisar" (fuera en el
sentido que suele ser un defecto): esas son las falsas alarmas reales. Usa solo data/processed/metricas_referencia.csv: no vuelve a pasar
el modelo.
"""

import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analizar_clip as ac  # noqa: E402


def dejar_uno_fuera(ref, fps):
    """Una fila por (golpe, vista, jugador, metrica) comparada. Funcion pura
    salvo por `comparar`, que tambien lo es."""
    filas = []
    for (golpe, vista), x in ref.groupby(["golpe", "vista"]):
        jugadores = x["jugador"].dropna().unique()
        if len(jugadores) <= ac.MIN_JUGADORES:
            continue  # sin el, quedarian menos de MIN_JUGADORES
        for jugador in jugadores:
            suyos = x[x["jugador"] == jugador]
            sesion = {m: suyos[m].median() for m in ac.FIABILIDAD if m in suyos}
            for f in ac.comparar(sesion, ref[ref["jugador"] != jugador], golpe, vista, fps):
                if "min" in f:
                    filas.append({"golpe": golpe, "vista": vista, "jugador": jugador,
                                  "metrica": f["metrica"], "valor": f["valor"],
                                  "min": f["min"], "max": f["max"],
                                  "dentro": f["estado"] == "dentro del rango",
                                  "a_revisar": f["valoracion"] == "a revisar"})
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--referencias",
                        default=os.path.join(ac.RAIZ, "data", "processed", "metricas_referencia.csv"))
    parser.add_argument("--fps", type=float, default=ac.FPS_FIABLE_BRAZO,
                        help="fps supuestos del usuario; por debajo de %(default)s "
                             "no se comparan las metricas del brazo")
    args = parser.parse_args()

    with warnings.catch_warnings():
        # metricas que no existen para un golpe (alcance en la derecha) dan
        # medianas de solo NaN: es lo esperado
        warnings.simplefilter("ignore", RuntimeWarning)
        t = dejar_uno_fuera(ac.cargar_referencias(args.referencias), args.fps)
    if t.empty:
        sys.exit("ERROR: ningun golpe y vista tiene jugadores suficientes")

    print("Profesional dentro del rango, dejandolo fuera de las referencias:\n")
    resumen = t.groupby("metrica")["dentro"].agg(["mean", "size"])
    resumen["fiabilidad"] = resumen.index.map(ac.FIABILIDAD)
    for metrica, r in resumen.sort_values("mean").iterrows():
        print(f"  {metrica:16s} {100 * r['mean']:5.0f} %  ({r['size']:2d} comparaciones, "
              f"fiabilidad {r['fiabilidad']})")
    print(f"\n  {'TOTAL':16s} {100 * t['dentro'].mean():5.0f} %  ({len(t)} comparaciones)")
    print(f"  {'falsas alarmas':16s} {100 * t['a_revisar'].mean():5.0f} %  "
          "(profesional marcado 'a revisar')")

    fuera = t[~t["dentro"]]
    if len(fuera):
        print("\nFuera del rango:")
        for _, f in fuera.iterrows():
            marca = "a revisar" if f["a_revisar"] else "sin problema"
            print(f"  {f['golpe']}/{f['vista']} {f['jugador']:14s} {f['metrica']:16s} "
                  f"{f['valor']:7.2f}  [{f['min']:.2f} - {f['max']:.2f}]  {marca}")


if __name__ == "__main__":
    main()
