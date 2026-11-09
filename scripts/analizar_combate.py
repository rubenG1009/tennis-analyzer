#!/usr/bin/env python3
"""Analiza un clip de boxeo o K-1 y dice que corregir primero, segun tu perfil.

    python scripts/analizar_combate.py data/raw/mi_sombra.mp4 \\
        --altura 178 --peso 76 --experiencia-meses 3 --postura ortodoxa \\
        --disciplina k1 --vista lateral

El perfil (altura, peso, experiencia, postura, flexibilidad, limitaciones) lo
pones tu: el video no lo sabe. Tambien se puede dar como JSON con --perfil.
Escribe <nombre>_combate.json y <nombre>_combate.md en --out-dir.

Pipeline: esqueleto (MediaPipe) -> correccion de inversiones -> golpes (puño y
patada) -> metricas en cm y m/s -> correcciones priorizadas.

Los umbrales son orientativos y no estan validados por un entrenador ni
calibrados con profesionales (ver combate_correcciones.py). El informe lo dice.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import combate_correcciones as cc  # noqa: E402
import combate_golpes as cg  # noqa: E402
import combate_metricas as cm  # noqa: E402
import combate_perfil as cp  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISTAS = ("lateral", "diagonal", "frontal")
AVISO = ("Umbrales orientativos: no están calibrados con profesionales ni validados por "
         "un entrenador. Son hipótesis para comprobar mirando el vídeo; si tu entrenador "
         "dice otra cosa, tiene razón él.")


MIN_FRAMES_VALIDOS_PCT = 80
CONSEJOS_GRABACION = (
    "Para que el análisis valga: cuerpo entero en plano (de la cabeza a los pies, sin cortes "
    "ni zoom), cámara fija a la altura de la cadera, de lado o en diagonal, un solo "
    "luchador, buena luz y fondo que contraste, y a 120-240 fps si puedes.")


MIN_VISIBLE_PCT = 60
NOMBRE_EXTREMO = {"left_wrist": "la muñeca izquierda", "right_wrist": "la muñeca derecha",
                  "left_ankle": "el tobillo izquierdo", "right_ankle": "el tobillo derecho"}


def visibles(vis):
    """% de fotogramas en que se ve cada muñeca y tobillo. Funcion pura."""
    return {n: float(np.mean(np.nan_to_num(np.asarray(vis[n], float), nan=0.0) >= 0.5) * 100)
            for n in NOMBRE_EXTREMO}


def calidad(sesion, n_golpes, visible=None):
    """Motivos por los que este clip no da para conclusiones (lista vacia = bien).
    Funcion pura."""
    motivos = []
    for n, pct in (visible or {}).items():
        if pct < MIN_VISIBLE_PCT:
            motivos.append(f"{NOMBRE_EXTREMO[n]} solo se ve en el {pct:.0f} % de los "
                           f"fotogramas (mínimo {MIN_VISIBLE_PCT} %): sus golpes se descartan")
    if n_golpes == 0:
        motivos.append("no se ha detectado ningún golpe")
        return motivos
    if sesion["frames_validos_pct"] < MIN_FRAMES_VALIDOS_PCT:
        motivos.append(f"solo se detecta a la persona en el {sesion['frames_validos_pct']:.0f} % "
                       f"de los fotogramas (mínimo {MIN_FRAMES_VALIDOS_PCT} %)")
    if sesion["fiables"] < cc.MIN_GOLPES:
        motivos.append(f"solo {sesion['fiables']} de {n_golpes} golpes detectados tienen el "
                       f"brazo o la pierna bien visibles (hacen falta {cc.MIN_GOLPES})")
    return motivos


def analizar_xy(xy, vis, fps, perfil, vista="lateral", saltos=None):
    """Del esqueleto (dict nombre -> (n, 2) en pixeles) al resultado completo.

    Funcion pura (sin disco ni video): es la que se testea.
    """
    golpes = cg.detectar(xy, fps, perfil.lado_delantero, saltos,
                         solo_horizontal=vista != "frontal", vis=vis)
    filas, sesion = cm.medir(xy, vis, fps, perfil, golpes)
    reglas = cc.evaluar(filas, sesion, perfil, vista)
    patadas = cc.altura_patadas(filas, perfil)
    return {
        "perfil": perfil.a_dict() | {"nivel": perfil.nivel,
                                     "cm_por_torso": round(perfil.cm_por_torso, 1)},
        "vista": vista, "fps": fps,
        "sesion": sesion, "golpes": filas, "reglas": reglas,
        "calidad": calidad(sesion, len(filas), visibles(vis)),
        "patadas": patadas,
        "notas": cc.notas_perfil(perfil, bool(patadas)) + (
            ["Vista frontal: los puños van hacia la cámara y apenas se detectan, y el juego de "
             "pies puede salir como patadas. Para puños y patadas, graba de lado o en diagonal."]
            if vista == "frontal" else []),
        "aviso": AVISO,
    }


def _num(x, dec=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dec}f}"


def informe_markdown(r, video=""):
    p, s = r["perfil"], r["sesion"]
    L = [f"# Análisis de combate{': ' + os.path.basename(video) if video else ''}", ""]
    L += [f"**Perfil:** {p['altura_cm']:.0f} cm, {p['peso_kg']:.0f} kg, "
          f"{p['experiencia_meses']} meses de práctica ({p['nivel']}), "
          f"postura {p['postura']}, disciplina {p['disciplina']}, vista {r['vista']}.", ""]
    L += [f"**Qué se ha visto:** {s['puños']} puños y {s['patadas']} patadas en "
          f"{s['duracion_s']:.0f} s ({s['combinaciones']} combinaciones, la más larga de "
          f"{s['combinacion_mas_larga']}). Guardia arriba el "
          f"{_num(s['guardia_arriba_pct'], 0)} % del tiempo. "
          f"Golpes con dato fiable: {s['fiables']} de {len(r['golpes'])}.", ""]
    if r["calidad"]:
        L += ["> **⚠ Este clip no da para conclusiones fiables:** "
              + "; ".join(r["calidad"]) + ". " + CONSEJOS_GRABACION, ""]
    for n in r["notas"]:
        L += [f"> {n}", ""]

    reglas = [x for x in r["reglas"] if x["estado"] in (cc.PRIORITARIO, cc.MEJORAR)]
    L += ["## Qué corregir, por orden", ""]
    if not reglas and r["calidad"]:
        L += ["No se puede decir qué corregir con este clip (ver el aviso de arriba).", ""]
    elif not reglas:
        L += ["Con los datos de este clip no sale nada claramente por corregir. "
              "No significa que sea perfecto: revisa el vídeo con tu entrenador.", ""]
    for i, x in enumerate(reglas[:cc.PRIMERAS], 1):
        u = x["unidad"]
        L += [f"### {i}. {x['problema'][0].upper()}{x['problema'][1:]}  ({x['estado']})", ""]
        base = "toda la sesión" if x["tipo"] == "sesion" else \
            f"mediana de {x['n']} golpes; {x['golpes_fuera']} fuera del objetivo"
        L += [f"- **Tu valor:** {_num(x['valor'])} {u} ({base}). **Objetivo para tu nivel:** "
              f"{_num(x['umbral_ok'])} {u}; prioritario a partir de {_num(x['umbral_mal'])}."]
        if "peor_t_s" in x:
            L += [f"- **Dónde mirar:** el golpe más claro está en el segundo {x['peor_t_s']}."]
        L += [f"- **Qué comprobar en el vídeo:** {x['que_mirar']}",
              "- **Causas habituales:** " + "; ".join(x["causas"]) + ".",
              "- **Ejercicios:** " + "; ".join(x["ejercicios"]) + ".", ""]
    resto = reglas[cc.PRIMERAS:]
    if resto:
        L += ["### Además, a vigilar", ""]
        L += [f"- {x['problema']} ({_num(x['valor'])} {x['unidad']})" for x in resto] + [""]

    bien = [x for x in r["reglas"] if x["estado"] == cc.BIEN]
    if bien:
        L += ["## Lo que va bien (dentro del objetivo de tu nivel)", ""]
        L += [f"- {x['metrica']}: {_num(x['valor'])} {x['unidad']} "
              f"(objetivo {_num(x['umbral_ok'])})" for x in bien] + [""]
    sin = [x for x in r["reglas"] if x["estado"] == "sin dato"]
    if sin:
        L += ["## Sin dato", ""] + [f"- {x['metrica']}: {x['motivo']}" for x in sin] + [""]

    if r["patadas"]:
        k = r["patadas"]
        z = k["zonas"]
        ref = k["referencia_cm"]
        L += ["## Altura de las patadas", "",
              f"{k['n']} patadas: {z['baja']} bajas, {z['media']} medias, {z['alta']} altas. "
              f"Máxima {k['altura_max_cm']:.0f} cm sobre el pie de apoyo (mediana "
              f"{k['altura_mediana_cm']:.0f} cm). Para tu altura, rodilla ≈ {ref['rodilla']} cm, "
              f"cadera ≈ {ref['cadera']} cm, pecho ≈ {ref['pecho']} cm, cabeza ≈ {ref['cabeza']} cm. "
              "No es un defecto: la zona que busques la decides tú con tu entrenador.", ""]

    L += ["## Cómo se ha medido", "",
          "Una cámara, en el plano de la imagen: lo que se dirige hacia la cámara se ve más "
          "corto y lento. La altura convierte las medidas del cuerpo a cm y m/s con "
          "proporciones medias de población (Winter): tu cuerpo concreto se aparta unos "
          "puntos. El peso solo da un «momento» (masa del segmento × velocidad) para comparar "
          "tus golpes entre sí; no se usa para estimar fuerza. No se distingue directo de "
          "gancho ni patada circular de frontal.", "", AVISO, ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--perfil", help="JSON con el perfil (alternativa a las opciones)")
    parser.add_argument("--altura", type=float, help="cm")
    parser.add_argument("--peso", type=float, help="kg")
    parser.add_argument("--experiencia-meses", type=int, default=0)
    parser.add_argument("--postura", choices=cp.POSTURAS, default="ortodoxa",
                        help="ortodoxa: pierna y mano izquierdas delante")
    parser.add_argument("--disciplina", choices=cp.DISCIPLINAS, default="k1")
    parser.add_argument("--flexibilidad", choices=cp.FLEXIBILIDADES, default="media")
    parser.add_argument("--edad", type=int)
    parser.add_argument("--limitacion", action="append", default=[],
                        help="repetible: 'rodilla derecha', 'hombro izquierdo'...")
    parser.add_argument("--vista", choices=VISTAS, default="lateral")
    parser.add_argument("--out-dir", default=os.path.join(RAIZ, "data", "processed"))
    parser.add_argument("--reusar", action="store_true",
                        help="No vuelve a extraer landmarks si ya existe el CSV")
    args = parser.parse_args()

    if args.perfil:
        perfil = cp.cargar(args.perfil)
    elif args.altura and args.peso:
        perfil = cp.Perfil(altura_cm=args.altura, peso_kg=args.peso,
                           experiencia_meses=args.experiencia_meses, postura=args.postura,
                           disciplina=args.disciplina, flexibilidad=args.flexibilidad,
                           edad=args.edad, limitaciones=args.limitacion)
    else:
        sys.exit("ERROR: indica --altura y --peso, o un --perfil JSON")

    # Imports pesados aqui para que las funciones puras se importen sin cv2/mediapipe.
    import corregir_esqueleto
    import extract_pose
    import metricas_golpe
    from evaluar_esqueleto import cargar, detectar_saltos_persona, longitud_torso

    base = os.path.splitext(os.path.basename(args.video))[0]
    pre = os.path.join(args.out_dir, base)
    if not (args.reusar and os.path.isfile(f"{pre}_landmarks.csv")):
        extract_pose.procesar_video(
            args.video, os.path.join(RAIZ, "models", "pose_landmarker_heavy.task"), args.out_dir)
    fps, ancho, alto = metricas_golpe.info_video(args.video)
    # La mano dominante solo la usa el ancla de raqueta, que aqui no hay; la
    # correccion se apoya en la continuidad del cuerpo entero.
    corregir_esqueleto.corregir(f"{pre}_landmarks.csv", ancho, alto, f"{pre}_corregido.csv",
                                None, f"{pre}_cortes.csv", perfil.lado_delantero)
    n, xy, vis = cargar(f"{pre}_corregido.csv", ancho, alto)
    saltos = detectar_saltos_persona(xy, longitud_torso(xy))
    if fps < 100:
        print(f"AVISO: {fps:.0f} fps. A esta velocidad el brazo y la pierna salen movidos en "
              "el golpe; graba a 120-240 fps si puedes.")

    r = analizar_xy(xy, vis, fps, perfil, args.vista, saltos)
    r["video"] = args.video
    ruta_json, ruta_md = f"{pre}_combate.json", f"{pre}_combate.md"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False, default=lambda o: None if np.isnan(o) else float(o))
    md = informe_markdown(r, args.video)
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\nInforme -> {ruta_md}\nDatos   -> {ruta_json}")


if __name__ == "__main__":
    main()
