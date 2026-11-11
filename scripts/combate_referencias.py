"""Rangos de referencia de profesionales para las metricas de combate.

Cada clip de un profesional da, por tipo de golpe, la mediana de cada metrica
sobre sus golpes fiables (si tiene al menos MIN_GOLPES). El rango de
referencia de una metrica es [minimo, maximo] de esas medianas entre clips,
ensanchado un MARGEN por cada lado (con pocos profesionales el rango real es
mas ancho que el observado) y solo existe con MIN_CLIPS clips. Mismo criterio
que la comparacion de tenis: la sesion de una persona se compara con lo que
hacen profesionales, no golpe a golpe.

Las metricas en cm se guardan en torsos (cm / cm_por_torso del perfil con el
que se midieron) para que valgan con cualquier altura, y se devuelven a cm con
el perfil de quien se compara.

AVISO: los clips de referencia son sombra y entrenamientos de YouTube, con
camaras y encuadres distintos; un profesional haciendo sombra no es un
profesional peleando. Es una referencia de orden de magnitud, no un estandar.
"""

import json

import numpy as np

MIN_GOLPES = 3
MIN_CLIPS = 3
MARGEN = 0.25
METRICAS = {
    "puño": ["guardia_otra_mano_cm", "retorno_guardia_ms", "inclinacion_deg", "giro_cadera_deg",
             "extension_pct", "velocidad_ms"],
    "patada": ["guardia_en_patada_cm", "retorno_pierna_ms", "inclinacion_atras_deg",
               "altura_pct", "velocidad_ms"],
}
EN_CM = {"guardia_otra_mano_cm", "guardia_en_patada_cm"}
# m/s depende de la altura igual que los cm: se guarda en torsos/s
EN_MS_ALTURA = {"velocidad_ms"}


def a_unidad_comun(metrica, valor, perfil):
    """Valor de un perfil concreto -> unidad independiente de la altura. Funcion pura."""
    if metrica in EN_CM:
        return valor / perfil.cm_por_torso
    if metrica in EN_MS_ALTURA:
        return valor / (perfil.cm_por_torso / 100)
    return valor


def de_unidad_comun(metrica, valor, perfil):
    if metrica in EN_CM:
        return valor * perfil.cm_por_torso
    if metrica in EN_MS_ALTURA:
        return valor * perfil.cm_por_torso / 100
    return valor


def medianas_clip(filas, perfil):
    """{tipo: {metrica: mediana en unidad comun}} de un clip. Funcion pura."""
    salida = {}
    for tipo, metricas in METRICAS.items():
        propios = [f for f in filas if f["tipo"] == tipo and f.get("fiable")]
        for m in metricas:
            vals = [a_unidad_comun(m, f[m], perfil) for f in propios
                    if f.get(m) is not None and not np.isnan(f[m])]
            if len(vals) >= MIN_GOLPES:
                salida.setdefault(tipo, {})[m] = float(np.median(vals))
    return salida


def construir(clips):
    """clips: {nombre: medianas_clip(...)} -> referencias {tipo: {metrica: {...}}}.
    Funcion pura."""
    ref = {}
    for tipo, metricas in METRICAS.items():
        for m in metricas:
            por_clip = {n: c[tipo][m] for n, c in clips.items() if m in c.get(tipo, {})}
            if len(por_clip) < MIN_CLIPS:
                continue
            v = np.array(list(por_clip.values()))
            ref.setdefault(tipo, {})[m] = {"min": float(v.min()), "max": float(v.max()),
                                           "mediana": float(np.median(v)), "n_clips": len(v),
                                           "clips": por_clip}
    return ref


def rango(ref, tipo, metrica, perfil):
    """(minimo, maximo, n_clips) en las unidades del perfil, con margen; None si
    no hay referencia. Funcion pura."""
    r = (ref or {}).get(tipo, {}).get(metrica)
    if not r:
        return None
    ancho = r["max"] - r["min"]
    lo, hi = r["min"] - MARGEN * ancho, r["max"] + MARGEN * ancho
    return (de_unidad_comun(metrica, lo, perfil), de_unidad_comun(metrica, hi, perfil), r["n_clips"])


def posicion(valor, rango_):
    """"dentro" / "por encima" / "por debajo" del rango. Funcion pura."""
    lo, hi, _ = rango_
    return "por debajo" if valor < lo else "por encima" if valor > hi else "dentro"


def guardar(ref, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
