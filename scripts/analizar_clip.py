#!/usr/bin/env python3
"""Pipeline completo de un clip: landmarks -> correccion -> golpes -> metricas
-> comparacion con los profesionales de referencia.

    python scripts/analizar_clip.py data/raw/mi_derecha.mp4 --golpe derecha --vista lateral

La comparacion es POR SESION, no por golpe: se toma la mediana de cada metrica
sobre todos los golpes del clip y se compara con el rango de los profesionales
de referencia del mismo tipo de golpe y vista (data/processed/
metricas_referencia.csv, generado con `metricas_golpe.py referencias`).

El rango es [minimo, maximo] de las medianas por jugador, ampliado
MARGEN_INTRA veces la dispersion tipica de un jugador (p10-p90 de sus golpes).
Se calibro dejando fuera a cada profesional y analizandolo como si fuera un
usuario: asi queda dentro del rango en ~88 % de las metricas. Comparar golpe a
golpe con el p10-p90 de todos los golpes daba 74 %, y con pocos jugadores de
referencia un solo jugador con muchos golpes dominaba el rango. Un valor
"fuera de rango" no significa que este mal: significa que ningun profesional de
referencia lo hace asi de forma habitual, y merece mirarse.

Cada metrica lleva su fiabilidad (FIABILIDAD), medida en los clips de
referencia (docs/hallazgos.md, 2026-10-07):

- alta: cambia poco si el impacto se desplaza 3 frames y no depende del brazo.
- baja: cambia mucho con 3 frames de error en el impacto, o el brazo sale mal
  colocado por el desenfoque a 30 fps. Solo se muestra comparacion si el video
  es de camara lenta (>= FPS_FIABLE_BRAZO).

Cada metrica comparada lleva ademas su valoracion: salirse del rango solo es
"a revisar" en el sentido que suele ser un defecto (poco giro de hombros, no
mucho), y entonces lleva el diagnostico de scripts/diagnosticos.py.

Escribe <nombre>_informe.json con todo, pensado para la etapa del LLM.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corregir_esqueleto  # noqa: E402
import diagnosticos  # noqa: E402
import extract_pose  # noqa: E402
import metricas_golpe  # noqa: E402
import segmentar_golpes  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS_FIABLE_BRAZO = 100
MIN_JUGADORES = 3
MARGEN_INTRA = 0.25
MIN_GOLPES_JUGADOR = 3
MIN_GOLPES_SESION = 3   # golpes con dato para dar el valor de la sesion

FIABILIDAD = {
    "rodilla_min": "alta",
    "tronco_impacto": "alta",
    "ancho_apoyos": "alta",
    "alcance": "alta",
    "altura_impacto": "media",
    "codo_impacto": "baja",
    "impacto_delante": "baja",
    "terminacion": "alta",
    "giro_hombros": "media",
    "peso_delante": "media",
}
METRICAS_BRAZO = {"codo_impacto", "impacto_delante", "altura_impacto"}

DESCRIPCION = {
    "rodilla_min": ("flexión máxima de rodillas en la carga", "grados (180 = pierna recta)"),
    "tronco_impacto": ("inclinación del tronco en el impacto", "grados respecto a la vertical"),
    "ancho_apoyos": ("separación de pies en el impacto", "longitudes de torso"),
    "alcance": ("altura de la muñeca sobre la cabeza en el impacto", "longitudes de torso"),
    "altura_impacto": ("altura del impacto", "0 = cadera, 1 = hombros"),
    "codo_impacto": ("extensión del codo en el impacto", "grados (180 = brazo recto)"),
    "impacto_delante": ("impacto por delante del cuerpo", "longitudes de torso"),
    "terminacion": ("altura de la muñeca al terminar el golpe", "0 = cadera, 1 = hombros"),
    "giro_hombros": ("amplitud del giro de hombros, de la preparación al final del giro",
                     "grados"),
    "peso_delante": ("cadera por delante de los pies en el impacto (transferencia del peso)",
                     "longitudes de torso"),
}


def dispersion(s):
    s = s.dropna()
    if len(s) < MIN_GOLPES_JUGADOR:
        return np.nan
    return float(s.quantile(0.9) - s.quantile(0.1))


def rango_referencia(ref, golpe, vista, metrica):
    """Rango de los profesionales para una metrica, o None si hay menos de
    MIN_JUGADORES con ese golpe y vista. `ref` necesita columna `jugador`.
    Funcion pura."""
    x = ref[(ref["golpe"] == golpe) & (ref["vista"] == vista)]
    medianas = x.groupby("jugador")[metrica].median().dropna()
    if len(medianas) < MIN_JUGADORES:
        return None
    intra = x.groupby("jugador")[metrica].agg(dispersion).median()
    intra = 0.0 if np.isnan(intra) else float(intra)
    return {"min": float(medianas.min() - MARGEN_INTRA * intra),
            "mediana": float(medianas.median()),
            "max": float(medianas.max() + MARGEN_INTRA * intra),
            "jugadores": int(len(medianas)), "golpes": int(x[metrica].notna().sum())}


def comparar(metricas, ref, golpe, vista, fps):
    """metricas: dict metrica -> valor de la sesion (mediana de los golpes)."""
    filas = []
    for clave, valor in metricas.items():
        if clave not in FIABILIDAD:
            continue
        fila = {"metrica": clave, "valor": None if pd.isna(valor) else float(valor),
                "fiabilidad": FIABILIDAD[clave], "descripcion": DESCRIPCION[clave][0],
                "unidad": DESCRIPCION[clave][1]}
        if fila["valor"] is None:
            fila["estado"] = ("sin dato (piernas fuera de plano o tapadas)"
                              if clave in metricas_golpe.METRICAS_PIERNAS
                              else "sin dato (brazo no visible)")
        elif clave in METRICAS_BRAZO and FIABILIDAD[clave] == "baja" and fps < FPS_FIABLE_BRAZO:
            fila["estado"] = f"no comparable a {fps:.0f} fps (graba en camara lenta)"
        else:
            r = rango_referencia(ref, golpe, vista, clave)
            if r is None:
                fila["estado"] = "sin referencias suficientes"
            else:
                fila.update(r)
                if valor < r["min"]:
                    fila["estado"] = "por debajo del rango"
                elif valor > r["max"]:
                    fila["estado"] = "por encima del rango"
                else:
                    fila["estado"] = "dentro del rango"
        fila["valoracion"] = diagnosticos.valoracion(clave, fila["estado"], golpe)
        fila["sentido_problema"] = diagnosticos.sentido_problema(clave, golpe)
        diag = diagnosticos.diagnostico(clave, fila["estado"], golpe)
        if diag:
            fila["diagnostico"] = diag
        filas.append(fila)
    return filas


def marcar_pocos_golpes(filas, medidos, total):
    """Anota en cuantos golpes se pudo medir cada metrica y, si fueron menos
    de MIN_GOLPES_SESION pero alguno, lo dice en el estado en vez de culpar a
    la visibilidad en general. Funcion pura."""
    for f in filas:
        f["golpes_medidos"] = medidos.get(f["metrica"], 0)
        if f["valor"] is None and 0 < f["golpes_medidos"] < MIN_GOLPES_SESION:
            f["estado"] = (f"sin dato (solo se pudo medir en {f['golpes_medidos']} "
                           f"de {total} golpes)")
    return filas


def cargar_referencias(ruta):
    ref = pd.read_csv(ruta)
    if "desplazamiento" in ref:
        ref = ref[ref["desplazamiento"].isna()]
    jugadores = pd.read_csv(os.path.join(RAIZ, "references", "manifest.csv")).set_index("nombre")["jugador"]
    return ref.assign(jugador=ref["clip"].map(jugadores))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--golpe", required=True,
                        choices=["derecha", "reves_2manos", "saque"],
                        help="Tipo de golpe del clip (de fondo no se clasifica solo)")
    parser.add_argument("--vista", required=True, choices=["lateral", "frontal", "espalda"])
    parser.add_argument("--lado", default="right", choices=["right", "left"],
                        help="Mano dominante (left para zurdos; se comparan con diestros "
                             "porque las metricas no cambian al espejar el cuerpo)")
    parser.add_argument("--out-dir", default=os.path.join(RAIZ, "data", "processed"))
    parser.add_argument("--referencias",
                        default=os.path.join(RAIZ, "data", "processed", "metricas_referencia.csv"))
    parser.add_argument("--excluir-jugador", metavar="NOMBRE",
                        help="Quita a ese jugador de las referencias (para validar con un pro)")
    parser.add_argument("--impactos", metavar="F1,F2,...",
                        help="Frames de impacto marcados a mano, en lugar de detectarlos. "
                             "Para sombras sin bola: el bucle de la preparacion es tan "
                             "rapido como el golpe y el segmentador no sabe donde esta el "
                             "contacto")
    parser.add_argument("--revision", action="store_true",
                        help="Genera tambien el informe visual (data/revision/<nombre>/index.html)")
    parser.add_argument("--reusar", action="store_true",
                        help="No vuelve a extraer landmarks si ya existe el CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.referencias):
        sys.exit("ERROR: faltan las referencias; ejecuta "
                 "`python scripts/metricas_golpe.py referencias`")
    # Zurdos: las referencias son de diestros, pero todas las metricas se miden
    # respecto al propio cuerpo y no cambian al espejarlo (comprobado: espejar
    # un clip de referencia y medirlo como zurdo da diferencia 0 en todas). Basta
    # con medir el lado dominante que toca.

    base = os.path.splitext(os.path.basename(args.video))[0]
    pre = os.path.join(args.out_dir, base)
    if not (args.reusar and os.path.isfile(f"{pre}_landmarks.csv")):
        extract_pose.procesar_video(
            args.video, os.path.join(RAIZ, "models", "pose_landmarker_heavy.task"),
            args.out_dir, os.path.join(RAIZ, "models", "efficientdet_lite2.tflite"))

    fps, ancho, alto = metricas_golpe.info_video(args.video)
    corregir_esqueleto.corregir(f"{pre}_landmarks.csv", ancho, alto, f"{pre}_corregido.csv",
                                f"{pre}_raqueta.csv", f"{pre}_cortes.csv", args.lado)
    quiero = "saque" if args.golpe == "saque" else "fondo"
    if args.impactos:
        golpes = [{"tipo": quiero, "impacto": int(f)} for f in args.impactos.split(",")]
    else:
        golpes = segmentar_golpes.segmentar(f"{pre}_corregido.csv", ancho, alto, fps,
                                            f"{pre}_raqueta.csv", f"{pre}_cortes.csv",
                                            args.lado, solo=quiero)
    descartados = [g for g in golpes if g["tipo"] != quiero]
    golpes = [g for g in golpes if g["tipo"] == quiero]

    ref = cargar_referencias(args.referencias)
    if args.excluir_jugador:
        ref = ref[ref["jugador"] != args.excluir_jugador]
    medidas = metricas_golpe.medir_golpes(f"{pre}_corregido.csv", ancho, alto, fps,
                                          [(g["impacto"], g["tipo"]) for g in golpes], args.lado)

    tabla = pd.DataFrame(medidas)
    medidos = {c: int(tabla[c].notna().sum()) for c in FIABILIDAD if c in tabla}
    sesion = {c: float(tabla[c].median()) if medidos[c] >= MIN_GOLPES_SESION else np.nan
              for c in medidos}
    informe = {
        "video": args.video, "golpe": args.golpe, "vista": args.vista, "fps": fps,
        "lado": args.lado,
        "impactos": "a mano" if args.impactos else "automaticos",
        "golpes_detectados": len(golpes),
        "golpes_descartados_por_tipo": [g["impacto"] for g in descartados],
        "sesion": marcar_pocos_golpes(comparar(sesion, ref, args.golpe, args.vista, fps),
                                      medidos, len(golpes)),
        "golpes": [{"impacto_frame": g["impacto"], "impacto_s": round(g["impacto"] / fps, 2),
                    **{k: (None if pd.isna(v) else float(v)) for k, v in m.items()}}
                   for g, m in zip(golpes, medidas)],
    }

    ruta = f"{pre}_informe.json"
    with open(ruta, "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    print()
    print(f"{len(golpes)} golpes de tipo {args.golpe} en {args.video} ({fps:.0f} fps), "
          f"impactos en s: {[g['impacto_s'] for g in informe['golpes']]}")
    print("\nSesion (mediana de los golpes) frente a los profesionales:")
    for m in informe["sesion"]:
        v = "-" if m["valor"] is None else f"{m['valor']:.2f}"
        rango = f" [pro: {m['min']:.2f} - {m['max']:.2f}]" if "min" in m else ""
        marca = "  <- a revisar" if m.get("valoracion") == "a revisar" else ""
        print(f"  {m['metrica']:16s} {v:>8s}{rango:28s} {m['estado']}  ({m['fiabilidad']}){marca}")
        if "diagnostico" in m:
            print(f"  {'':16s} {m['diagnostico']['problema']}")
    print(f"\nInforme -> {ruta}")

    if args.revision:
        import revision_html
        clip, video, procesado, cajas, corr = revision_html.desde_informe(ruta)
        destino = revision_html.escribir_clip(
            clip, video, procesado, cajas, corr,
            ref_dirs=(os.path.join(RAIZ, "data", "processed", "ref_v2"),
                      os.path.join(RAIZ, "data", "raw", "ref")))
        revision_html.escribir_indice()
        print(f"Revision -> {os.path.join(destino, 'index.html')}")


if __name__ == "__main__":
    main()
