"""Metricas de cada golpe y de la sesion (boxeo / K-1), en unidades reales.

Todo se mide en el plano de la imagen con una camara, relativo al propio
cuerpo (torsos) y se convierte a cm y m/s con el perfil (altura). Son
aproximaciones: una patada o un puño que se dirigen hacia la camara se ven mas
cortos y mas lentos de lo que son. Cada golpe lleva `fiable`: False si el
extremo o su cadena (hombro-codo-muneca / cadera-rodilla-tobillo) no se veia, y
entonces sus metricas no cuentan.

Puño: velocidad_ms, extension_pct, guardia_otra_mano_cm (cuanto baja la mano
que no golpea respecto a la nariz), retorno_guardia_ms, giro_cadera_deg,
giro_hombros_deg, inclinacion_deg (+ adelante, - atras), momento_kgms.

Patada: velocidad_ms, altura_cm, altura_pct, zona (baja/media/alta),
retorno_pierna_ms, inclinacion_atras_deg, guardia_en_patada_cm, momento_kgms.

El momento es masa del segmento (brazo o pierna segun el peso del perfil) x
velocidad: solo sirve para comparar tus golpes entre si y tu evolucion.
"""

import numpy as np

from combate_golpes import (BRAZO_EN_TORSOS, CADENA, UMBRAL_VIS, combinaciones,
                            levante_reposo, torso_suave, visibilidad)
from evaluar_esqueleto import punto_medio
from metricas_golpe import angulo, angulo_fuera_del_plano

RETORNO_MAX_S = 1.5       # si no vuelve a la guardia en este tiempo, se da este valor
TOLERANCIA_EXTENSION = 0.15  # del brazo estirado sobre la extension de reposo para dar el puño por recogido
LEVANTE_RECUPERADO = 0.3  # torsos de levante bajo los que la pierna esta recogida
GUARDIA_ALTA_MAX = 0.7    # torsos bajo la nariz: mas abajo, la mano esta fuera de guardia
ZONA_BAJA_MAX = 0.50      # fraccion de la altura: la patada llega al muslo
ZONA_MEDIA_MAX = 0.78     # ... al torso; por encima, a la cabeza
TIPO_PUÑO, TIPO_PATADA = "puño", "patada"
CODO_RECTO = 150          # grados: con el codo casi estirado es un recto (jab o cruzado)
UPPERCUT_MIN = 0.3        # torsos que la muñeca debe subir (y mas que avanzar) para ser uppercut


def clasificar_puño(codo_deg, subida, avance, rol, vista):
    """Nombre del puño: jab/cruzado (recto), gancho o uppercut, con su lado.

    Con el codo casi estirado es un recto (delantero = jab, trasero = cruzado).
    Con el codo doblado, es uppercut si la muñeca sube mas de UPPERCUT_MIN
    torsos y mas de lo que avanza, y gancho si no. Subida y avance en torsos.
    En vista frontal el codo no se ve (el brazo va hacia la camara): None.
    Funcion pura.
    """
    if vista == "frontal" or codo_deg is None or np.isnan(codo_deg):
        return None, None
    lado = "adelantado" if rol == "delantera" else "trasero"
    if codo_deg >= CODO_RECTO:
        return "recto", "jab" if rol == "delantera" else "cruzado"
    if subida > UPPERCUT_MIN and subida > avance:
        return "uppercut", f"uppercut {lado}"
    return "gancho", f"gancho {lado}"


def zona_patada(altura_pct):
    """baja / media / alta segun la altura del pie respecto al de apoyo, en %
    de la altura de la persona. Funcion pura."""
    if altura_pct < ZONA_BAJA_MAX * 100:
        return "baja"
    return "media" if altura_pct < ZONA_MEDIA_MAX * 100 else "alta"


def inclinacion_firmada(hombros, cadera, direccion):
    """Inclinacion del tronco con la vertical en grados, con signo respecto a
    la direccion del golpe: positiva = hombros por delante de la cadera
    (adelante), negativa = por detras (echado atras). Funcion pura sobre
    puntos (2,)."""
    d = np.asarray(hombros) - np.asarray(cadera)
    return float(np.degrees(np.arctan2(d[0] * direccion, -d[1])))


def _med(serie, a, b):
    v = serie[max(a, 0):b + 1]
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def _pct(serie, a, b, q=75):
    v = serie[max(a, 0):b + 1]
    v = v[~np.isnan(v)]
    return float(np.percentile(v, q)) if len(v) else np.nan


def preparar(xy, fps):
    """Series por frame que comparten todos los golpes."""
    torso = torso_suave(xy)
    cadera = punto_medio(xy["left_hip"], xy["right_hip"])
    hombros = punto_medio(xy["left_shoulder"], xy["right_shoulder"])
    s = {"torso": torso, "cadera": cadera, "hombros": hombros, "fps": fps}
    for lado in ("left", "right"):
        s[f"guardia_{lado}"] = (xy[f"{lado}_wrist"][:, 1] - xy["nose"][:, 1]) / torso
        v = np.full(len(torso), np.nan)
        rel = (xy[f"{lado}_wrist"] - cadera) / torso[:, None]
        v[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        s[f"vel_muneca_{lado}"] = v
        va = np.full(len(torso), np.nan)
        rel = (xy[f"{lado}_ankle"] - cadera) / torso[:, None]
        va[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        s[f"vel_tobillo_{lado}"] = va
    s["giro_cadera"] = angulo_fuera_del_plano(xy["left_hip"], xy["right_hip"], torso)
    s["giro_hombros"] = angulo_fuera_del_plano(xy["left_shoulder"], xy["right_shoulder"], torso)
    return s


def _retorno(serie_ok, desde, fps):
    """ms desde `desde` hasta el primer frame con serie_ok True; tope si no hay."""
    tope = int(RETORNO_MAX_S * fps)
    for k in range(tope):
        if desde + k >= len(serie_ok):
            break
        if serie_ok[desde + k]:
            return k / fps * 1000
    return RETORNO_MAX_S * 1000


def medir_golpe(g, xy, vis, s, perfil, vista="lateral"):
    fps, torso = s["fps"], s["torso"]
    lado = g["lado"]
    otro = "left" if lado == "right" else "right"
    cm = perfil.cm_por_torso
    a, b, p, ref = g["inicio"], g["fin"], g["pico"], g["ref"]
    fila = {k: g[k] for k in ("tipo", "lado", "rol", "pico", "t_s")}

    fila["fiable"] = visibilidad(vis, CADENA[g["tipo"]](lado), a, b) >= UMBRAL_VIS

    v_pico = np.nanmax(s[f"vel_{'muneca' if g['tipo'] == TIPO_PUÑO else 'tobillo'}_{lado}"]
                       [max(p - 2, 0):p + 3])
    fila["velocidad_ms"] = float(v_pico * cm / 100)
    masa = perfil.masa_brazo_kg if g["tipo"] == TIPO_PUÑO else perfil.masa_pierna_kg
    fila["momento_kgms"] = float(masa * fila["velocidad_ms"])

    # Inclinacion del tronco con signo respecto a la direccion del golpe.
    extremo = xy[f"{lado}_{'wrist' if g['tipo'] == TIPO_PUÑO else 'ankle'}"]
    direccion = np.sign(extremo[ref, 0] - s["cadera"][ref, 0]) or 1.0
    fila["inclinacion_deg"] = inclinacion_firmada(s["hombros"][ref], s["cadera"][ref], direccion)

    if g["tipo"] == TIPO_PUÑO:
        muneca = xy[f"{lado}_wrist"]
        dist = np.linalg.norm(muneca - xy[f"{lado}_shoulder"], axis=1) / (torso * BRAZO_EN_TORSOS)
        fila["extension_pct"] = float(np.nanmax(dist[a:b + 1]) * 100)
        fila["guardia_otra_mano_cm"] = _pct(s[f"guardia_{otro}"], a, b) * cm
        # Vuelve a la guardia cuando el brazo se recoge: la extension baja a la
        # de reposo (mediana del clip) mas una tolerancia. No sirve mirar solo la
        # altura y la velocidad: en la extension maxima la mano esta casi parada.
        ext = np.linalg.norm(muneca - xy[f"{lado}_shoulder"], axis=1) / (torso * BRAZO_EN_TORSOS)
        vuelto = np.nan_to_num(ext, nan=9e9) <= np.nanmedian(ext) + TOLERANCIA_EXTENSION
        fila["retorno_guardia_ms"] = _retorno(vuelto, ref, fps)
        codo = angulo(xy[f"{lado}_shoulder"], xy[f"{lado}_elbow"], muneca)
        fila["codo_deg"] = _med(codo, ref - 1, ref + 1)
        subida = (muneca[a, 1] - muneca[ref, 1]) / torso[ref]
        avance = abs(muneca[ref, 0] - muneca[a, 0]) / torso[ref]
        fila["variante"], fila["nombre"] = clasificar_puño(fila["codo_deg"], subida, avance,
                                                         g["rol"], vista)
        # el giro se mide con la cadera y los hombros: si no se ven, no hay dato
        vis_giro = visibilidad(vis, ["left_hip", "right_hip", "left_shoulder", "right_shoulder"], a, b)
        for clave, serie in (("giro_cadera_deg", "giro_cadera"), ("giro_hombros_deg", "giro_hombros")):
            v = s[serie][a:b + 1]
            fila[clave] = float(np.nanmax(v) - np.nanmin(v)) \
                if vis_giro >= UMBRAL_VIS and not np.isnan(v).all() else np.nan
    else:
        levante = levante_reposo(xy, lado, torso)
        fila["altura_cm"] = float(levante[ref] * cm)
        fila["altura_pct"] = float(fila["altura_cm"] / perfil.altura_cm * 100)
        fila["zona"] = zona_patada(fila["altura_pct"])
        recogida = np.nan_to_num(levante, nan=9e9) <= LEVANTE_RECUPERADO
        fila["retorno_pierna_ms"] = _retorno(recogida, ref, fps)
        fila["inclinacion_atras_deg"] = -fila["inclinacion_deg"]
        peor = np.fmax(s["guardia_left"], s["guardia_right"])
        fila["guardia_en_patada_cm"] = _pct(peor, a, b) * cm
    if not fila["fiable"]:
        for k, v in list(fila.items()):
            if k not in ("tipo", "lado", "rol", "pico", "t_s", "fiable", "zona", "variante", "nombre"):
                fila[k] = np.nan
    return fila


def medir_sesion(golpes, filas, xy, s, fps):
    """Metricas de toda la sesion: % de tiempo con la guardia arriba, ritmo y
    combinaciones. Un frame cuenta como guardia arriba si las dos manos estan
    por encima de GUARDIA_ALTA_MAX torsos bajo la nariz, o si la mano que no
    lo esta esta golpeando en ese momento."""
    n = len(s["torso"])
    ocupada = {"left": np.zeros(n, bool), "right": np.zeros(n, bool)}
    for g in golpes:
        if g["tipo"] == TIPO_PUÑO:
            ocupada[g["lado"]][g["inicio"]:g["fin"] + 1] = True
    ok = np.ones(n, bool)
    valido = ~np.isnan(s["torso"])
    for lado in ("left", "right"):
        g = s[f"guardia_{lado}"]
        ok &= (np.nan_to_num(g, nan=9e9) <= GUARDIA_ALTA_MAX) | ocupada[lado]
    guardia_pct = float(ok[valido].mean() * 100) if valido.any() else np.nan
    minutos = valido.sum() / fps / 60
    grupos = combinaciones(golpes, fps)
    nombres = {}
    for f in filas:
        if f.get("nombre"):
            nombres[f["nombre"]] = nombres.get(f["nombre"], 0) + 1
    return {
        "tipos_de_puño": nombres,
        "frames_validos_pct": float(valido.mean() * 100),
        "duracion_s": float(n / fps),
        "guardia_arriba_pct": guardia_pct,
        "puños": sum(g["tipo"] == TIPO_PUÑO for g in golpes),
        "patadas": sum(g["tipo"] == TIPO_PATADA for g in golpes),
        "golpes_por_min": float(len(golpes) / minutos) if minutos else np.nan,
        "combinaciones": len(grupos),
        "combinacion_mas_larga": max((len(c) for c in grupos), default=0),
        "fiables": int(sum(f["fiable"] for f in filas)),
    }


def medir(xy, vis, fps, perfil, golpes, vista="lateral"):
    s = preparar(xy, fps)
    filas = [medir_golpe(g, xy, vis, s, perfil, vista) for g in golpes]
    return filas, medir_sesion(golpes, filas, xy, s, fps)
