"""Detecta puños y patadas en un esqueleto corregido (boxeo / K-1).

Un golpe es un pico de velocidad del extremo (muneca para el puño, tobillo
para la patada) relativa a la cadera y en torsos, como en segmentar_golpes.py,
con un filtro que separa el golpe del movimiento que no lo es:

- Puño: el brazo tiene que extenderse. Un ajuste de guardia mueve la muneca
  deprisa pero no la aleja del hombro. Extension = distancia hombro-muneca
  frente a la longitud anatomica del brazo (en torsos, proporciones de Winter),
  no frente al propio clip: en un clip sin puños el maximo del clip seria un
  ajuste de guardia y pasaria por golpe.
  Ademas, la extension tiene que ser sobre todo horizontal: una mano que cae
  deprisa desde la guardia (el defecto que se quiere medir) aleja la muneca
  del hombro hacia abajo y no es un golpe. En vista frontal el puño va hacia la
  camara y esto no se puede exigir (solo_horizontal=False), asi que ahi la
  deteccion de puños es menos fiable.
- Patada: el tobillo tiene que subir por encima del de apoyo. Un paso o un
  salto de footwork mueve el pie rapido pero apenas lo levanta.

Se distingue mano o pierna delantera y trasera segun la postura del perfil,
pero NO directo frente a gancho, ni patada circular frente a frontal: con una
camara en el plano de la imagen no se ve la trayectoria en profundidad.

Los umbrales son orientativos, sin calibrar con clips de referencia de K-1.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from combate_perfil import FRAC_BRAZO, FRAC_TORSO
from evaluar_esqueleto import longitud_torso, punto_medio
from segmentar_golpes import picos_de_golpe, velocidad_relativa

SIGMA_SUAVIZADO = 1.5       # frames (a 240 fps son 6 ms; a 30 fps, 50 ms)
PROMINENCIA_RELATIVA = 3.0  # veces la mediana de la velocidad
SEPARACION_MIN_S = 0.25     # entre dos golpes del mismo extremo
FRACCION_FIN = 0.2          # el golpe acaba cuando la velocidad cae a esta fraccion del pico
TOPE_VENTANA_S = 0.6        # segundos a cada lado del pico, como maximo
EXTENSION_MIN = 0.6         # fraccion del brazo estirado que debe alcanzar el puño
BRAZO_EN_TORSOS = FRAC_BRAZO / FRAC_TORSO  # hombro-muneca con el brazo estirado, en torsos
FRACCION_HORIZONTAL = 0.7   # del minimo de extension debe ser horizontal (ver detectar)
LEVANTE_MIN = 0.5           # torsos que debe subir el tobillo sobre el de apoyo
UMBRAL_VIS = 0.5            # visibilidad minima de la cadena del extremo durante el golpe
MARGEN_SOLAPE_S = 0.1       # un puño dentro de una patada (mas este margen) no cuenta
SEPARACION_MANOS_S = 0.15   # dos puños de manos opuestas mas cerca que esto: uno solo
TIPOS = ("puño", "patada")
CADENA = {"puño": lambda lado: [f"{lado}_shoulder", f"{lado}_elbow", f"{lado}_wrist"],
          "patada": lambda lado: [f"{lado}_hip", f"{lado}_knee", f"{lado}_ankle"]}


def torso_suave(xy):
    import pandas as pd
    return pd.Series(longitud_torso(xy)).rolling(15, center=True, min_periods=1).median().to_numpy()


def visibilidad(vis, nombres, a, b):
    """Visibilidad de la cadena `nombres` entre los frames a y b (ambos
    incluidos): la del landmark menos visible, cada uno por su mediana. Una
    cadena (hombro-codo-muneca) es tan fiable como su punto peor visto; los
    frames sin deteccion cuentan 0. Funcion pura."""
    medianas = [float(np.median(np.nan_to_num(np.asarray(vis[n][max(a, 0):b + 1], float), nan=0.0)))
                for n in nombres]
    return min(medianas) if medianas else 0.0


def levante_reposo(xy, lado, torso):
    """Cuanto sube el tobillo `lado` sobre el otro, en torsos, restando la
    diferencia de reposo (mediana del clip): con la camara algo inclinada o los
    pies a distinta profundidad, uno parece siempre mas alto que el otro. Funcion pura."""
    otro = "left" if lado == "right" else "right"
    bruto = (xy[f"{otro}_ankle"][:, 1] - xy[f"{lado}_ankle"][:, 1]) / torso
    return bruto - np.nanmedian(bruto)


def ventana(s, p, tope, fraccion=FRACCION_FIN):
    """(inicio, fin) del golpe con pico en p: hasta que s cae por debajo de
    fraccion * s[p], con tope de frames. Funcion pura."""
    umbral = fraccion * s[p]
    a = p
    while a > 0 and s[a - 1] > umbral and p - a < tope:
        a -= 1
    b = p
    while b < len(s) - 1 and s[b + 1] > umbral and b - p < tope:
        b += 1
    return a, b


def detectar(xy, fps, lado_delantero, saltos=None, solo_horizontal=True, vis=None,
             detectar_patadas=True):
    """Lista de golpes ordenada por frame.

    Cada golpe: tipo ("puño" | "patada"), lado ("left" | "right"), rol
    ("delantera" | "trasera"), inicio, pico (maxima velocidad), ref (frame de
    referencia: maxima extension del puño o maximo levante de la patada), fin
    y t_s; los puños llevan ademas `extension` (fraccion del brazo estirado).
    `saltos`: mascara de frames a ignorar (cortes, otra persona). Con `vis`
    (dict nombre -> visibilidad por frame) se descartan los golpes cuyo
    extremo no se veia: con la cadera o las piernas fuera de plano el
    esqueleto inventa patadas.

    Dos filtros mas, de lo que se vio en clips reales:
    - un puño dentro de una patada se descarta: son los brazos haciendo de
      contrapeso, no un golpe (en K-1 el puño y la patada van seguidos);
    - dos puños de manos opuestas a la vez son uno solo: el de mayor extension.
    """
    torso = torso_suave(xy)
    cadera = punto_medio(xy["left_hip"], xy["right_hip"])
    validos = ~np.isnan(torso)
    if saltos is not None:
        validos &= ~saltos
    separacion = max(1, int(SEPARACION_MIN_S * fps))
    tope = max(2, int(TOPE_VENTANA_S * fps))
    golpes = []

    for lado in ("left", "right"):
        otro = "left" if lado == "right" else "right"
        rol = "delantera" if lado == lado_delantero else "trasera"

        for tipo, extremo in (("puño", f"{lado}_wrist"), ("patada", f"{lado}_ankle")):
            if tipo == "patada" and not detectar_patadas:
                continue
            punto = xy[extremo]
            v = velocidad_relativa(punto, cadera, torso)
            ok = validos & ~np.isnan(v)
            s = gaussian_filter1d(np.nan_to_num(v), SIGMA_SUAVIZADO)
            if tipo == "patada":
                levante = levante_reposo(xy, lado, torso)
            if tipo == "puño":
                brazo = torso * BRAZO_EN_TORSOS
                dist = np.linalg.norm(punto - xy[f"{lado}_shoulder"], axis=1) / brazo
                horiz = np.abs(punto[:, 0] - xy[f"{lado}_shoulder"][:, 0]) / brazo
            for p in picos_de_golpe(v, ok, separacion, PROMINENCIA_RELATIVA):
                a, b = ventana(s, p, tope)
                if vis is not None and visibilidad(
                        vis, CADENA[tipo](lado), a, b) < UMBRAL_VIS:
                    continue
                if tipo == "puño":
                    tramo = dist[a:b + 1]
                    if np.isnan(tramo).all() or np.nanmax(tramo) < EXTENSION_MIN:
                        continue
                    ref = a + int(np.nanargmax(tramo))
                    if solo_horizontal and horiz[ref] < FRACCION_HORIZONTAL * EXTENSION_MIN:
                        continue
                else:
                    tramo = levante[a:b + 1]
                    if np.isnan(tramo).all() or np.nanmax(tramo) < LEVANTE_MIN:
                        continue
                    ref = a + int(np.nanargmax(tramo))
                g = {"tipo": tipo, "lado": lado, "rol": rol, "inicio": int(a),
                     "pico": int(p), "ref": int(ref), "fin": int(b), "t_s": round(p / fps, 2)}
                if tipo == "puño":
                    g["extension"] = float(np.nanmax(tramo))
                golpes.append(g)
    return filtrar_solapes(sorted(golpes, key=lambda g: g["pico"]), fps)


def filtrar_solapes(golpes, fps):
    """Quita los puños que son contrapeso de una patada y deja uno solo de cada
    par de puños simultaneos de manos opuestas. Funcion pura."""
    margen = int(MARGEN_SOLAPE_S * fps)
    patadas = [g for g in golpes if g["tipo"] == "patada"]
    sin_contrapeso = [g for g in golpes if g["tipo"] == "patada" or not any(
        k["inicio"] - margen <= g["pico"] <= k["fin"] + margen for k in patadas)]
    cerca = int(SEPARACION_MANOS_S * fps)
    quitar = set()
    puños = [(i, g) for i, g in enumerate(sin_contrapeso) if g["tipo"] == "puño"]
    for (i, a), (j, b) in zip(puños, puños[1:]):
        if a["lado"] != b["lado"] and abs(a["pico"] - b["pico"]) <= cerca:
            quitar.add(i if a["extension"] < b["extension"] else j)
    return [g for i, g in enumerate(sin_contrapeso) if i not in quitar]


def combinaciones(golpes, fps, separacion_s=0.8):
    """Agrupa golpes seguidos (menos de separacion_s entre uno y el siguiente).

    Devuelve lista de listas de indices. Un golpe suelto es una combinacion de
    uno. Funcion pura.
    """
    grupos = []
    for i, g in enumerate(golpes):
        if grupos and (g["pico"] - golpes[grupos[-1][-1]]["pico"]) / fps <= separacion_s:
            grupos[-1].append(i)
        else:
            grupos.append([i])
    return grupos
