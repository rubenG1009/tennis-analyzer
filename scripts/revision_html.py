#!/usr/bin/env python3
"""Informe visual de revision de cada clip: se abre con doble clic, sin servidor.

    # un clip propio (el _informe.json que escribe analizar_clip.py)
    python scripts/revision_html.py data/processed/mio/mi_derecha_240_informe.json

    # los 21 clips de referencia, con cada profesional comparado con los demas
    python scripts/revision_html.py --referencias

Escribe data/revision/<clip>/index.html y data/revision/index.html con todos
los clips revisados. data/ no esta en el repositorio: el informe lleva el
video de la persona y se queda en local.

Para que sirve: para ver si el esqueleto esta bien puesto, que corrigio
corregir_esqueleto y donde, y si los impactos son impactos. Casi todos los
fallos del pipeline se encontraron mirando frames; esto es mirarlos sin
fabricar tiras de imagenes a mano. Por clip:

- ficha del video y avisos automaticos (deteccion, visibilidad, que no se
  pudo medir y por que);
- el video con el esqueleto CORREGIDO dibujado encima en el navegador, el
  original donde difiere y las articulaciones corregidas resaltadas;
- linea de tiempo clicable: deteccion, visibilidad del brazo y las piernas,
  volteos, picos, cortes de escena, raqueta y golpes;
- una ficha por golpe con preparacion, impacto y final del giro;
- la sesion frente al rango de los profesionales, con su diagnostico.

El video de depuracion de extract_pose no vale aqui: esta en mp4v, que no
reproduce ningun navegador, y lleva el esqueleto sin corregir. Se recodifica
el original a H.264 con el ffmpeg de imageio-ffmpeg.
"""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analizar_clip as ac  # noqa: E402
import metricas_golpe as mg  # noqa: E402
from evaluar_esqueleto import cargar  # noqa: E402
from extract_pose import CONEXIONES, NOMBRES_LANDMARKS  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_REVISION = os.path.join(RAIZ, "data", "revision")
PLANTILLA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision", "plantilla.html")

# Cara reducida a nariz y orejas: el resto no aporta a la revision y pesa.
PUNTOS = [0, 7, 8] + list(range(11, 17)) + list(range(23, 33))
HUESOS = [(PUNTOS.index(a), PUNTOS.index(b)) for a, b in CONEXIONES
          if a in PUNTOS and b in PUNTOS]
ESCALA = 10000          # coordenadas normalizadas como enteros 0-10000
LADO_VIDEO = 960        # lado mayor del video recodificado, en pixeles
PUNTOS_LINEA = 1200     # muestras por serie en la linea de tiempo


def tramos(mascara):
    """[[inicio, fin], ...] (fin incluido) de los True contiguos. Funcion pura."""
    m = np.asarray(mascara, dtype=bool)
    bordes = np.diff(np.r_[0, m.astype(int), 0])
    return [[int(a), int(b) - 1] for a, b in zip(np.flatnonzero(bordes == 1),
                                                 np.flatnonzero(bordes == -1))]


def reducir(serie, n=PUNTOS_LINEA):
    """Serie (n_frames,) -> como mucho n valores, el minimo de cada bloque (para
    la visibilidad lo que importa es si en algun frame cae). NaN -> None.
    Funcion pura."""
    v = np.asarray(serie, dtype=float)
    if len(v) == 0:
        return []
    bloques = np.array_split(v, min(n, len(v)))
    salida = []
    for b in bloques:
        b = b[np.isfinite(b)]
        salida.append(round(float(b.min()), 2) if len(b) else None)
    return salida


def leer_correcciones(ruta_corregido):
    """DataFrame de (frame, landmark_id, correccion) con correccion no vacia."""
    df = pd.read_csv(ruta_corregido, usecols=["frame", "landmark_id", "correccion"])
    return df[df["correccion"].notna() & (df["correccion"] != "")]


def eventos_de_correccion(corr):
    """Tramos de frames con volteo y lista de frames con pico. Funcion pura."""
    volteos = sorted(set(corr.loc[corr["correccion"] == "volteo", "frame"].astype(int)))
    picos = sorted(set(corr.loc[corr["correccion"] == "pico", "frame"].astype(int)))
    marca = np.zeros(max(volteos) + 1 if volteos else 0, bool)
    marca[volteos] = True
    return tramos(marca), picos


def esqueleto(ruta_csv, n):
    """Lista por punto de PUNTOS de [x, y] enteros por frame (None si falta)."""
    df = pd.read_csv(ruta_csv, usecols=["frame", "landmark_id", "x", "y"])
    df = df[df["landmark_id"].isin(PUNTOS)]
    salida = []
    for p in PUNTOS:
        g = df[df["landmark_id"] == p].set_index("frame").reindex(range(n))
        xs = np.round(g["x"].to_numpy() * ESCALA)
        ys = np.round(g["y"].to_numpy() * ESCALA)
        salida.append([None if np.isnan(x) else [int(x), int(y)] for x, y in zip(xs, ys)])
    return salida


def originales_distintos(ruta_landmarks, corr):
    """{frame: {indice en PUNTOS: [x, y]}} del modelo SIN corregir, solo donde
    el corrector cambio algo: con eso el navegador dibuja el original encima."""
    if corr.empty:
        return {}
    df = pd.read_csv(ruta_landmarks, usecols=["frame", "landmark_id", "x", "y"])
    tocados = set(map(tuple, corr[["frame", "landmark_id"]].astype(int).to_numpy()))
    frames = sorted({f for f, _ in tocados})
    df = df[df["frame"].isin(frames) & df["landmark_id"].isin(PUNTOS)].dropna()
    salida = {}
    for f, lid, x, y in df[["frame", "landmark_id", "x", "y"]].itertuples(index=False):
        salida.setdefault(int(f), {})[PUNTOS.index(int(lid))] = [
            int(round(x * ESCALA)), int(round(y * ESCALA))]
    return salida


def raqueta(ruta, n):
    """Tramos de frames con raqueta detectada y su caja por frame."""
    if not os.path.isfile(ruta):
        return [], {}
    df = pd.read_csv(ruta)
    if df.empty:
        return [], {}
    df = df.sort_values("score", ascending=False).drop_duplicates("frame")
    marca = np.zeros(n, bool)
    marca[df["frame"].astype(int).clip(0, n - 1)] = True
    cajas = {int(r.frame): [int(round(v * ESCALA)) for v in (r.x0, r.y0, r.x1, r.y1)]
             for r in df.itertuples(index=False)}
    return tramos(marca), cajas


def avisos(clip):
    """Frases de lo que hay que saber antes de fiarse del clip. Funcion pura
    sobre el dict que construye datos_clip."""
    a = []
    pct = clip["deteccion_pct"]
    if pct < 90:
        a.append(f"Solo se detecta a la persona en el {pct:.0f} % de los frames.")
    for nombre, v in clip["visible_pct"].items():
        if v < 50:
            a.append(f"{nombre.capitalize()}: visible solo en el {v:.0f} % de los frames.")
    if clip["fps"] < ac.FPS_FIABLE_BRAZO:
        a.append(f"Grabado a {clip['fps']:.0f} fps: el codo y el impacto por delante no se "
                 f"comparan por debajo de {ac.FPS_FIABLE_BRAZO} fps.")
    if clip.get("impactos") == "a mano":
        a.append("Impactos marcados a mano (sombras sin bola: el pico de velocidad de la "
                 "muneca cae en la preparacion).")
    n = len(clip["golpes"])
    if clip.get("con_golpes") and n < 5:
        a.append(f"Solo {n} golpes: las conclusiones de la sesion son provisionales.")
    for m in clip.get("sesion") or []:
        if m["estado"].startswith("sin dato"):
            a.append(f"{m['metrica']}: {m['estado']}.")
    return a


def recodificar(origen, destino):
    """Video H.264 sin audio, lado mayor LADO_VIDEO, mismos frames que el
    original (passthrough: el esqueleto se sincroniza por indice de frame)."""
    import imageio_ffmpeg
    escala = (f"scale='if(gt(iw,ih),{LADO_VIDEO},-2)':'if(gt(iw,ih),-2,{LADO_VIDEO})'")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", origen,
           "-an", "-vf", f"{escala},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "27", "-fps_mode", "passthrough", "-movflags", "+faststart", destino]
    subprocess.run(cmd, check=True)


def miniatura(cap, frame, puntos, destino, alto=360):
    """JPG del frame con el esqueleto corregido dibujado."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    if not ok:
        return False
    h, w = im.shape[:2]
    k = alto / h
    im = cv2.resize(im, (int(w * k), alto))
    pts = [puntos[i][frame] for i in range(len(PUNTOS))]
    px = [None if p is None else (int(p[0] / ESCALA * w * k), int(p[1] / ESCALA * h * k))
          for p in pts]
    for a, b in HUESOS:
        if px[a] and px[b]:
            izq = "left" in NOMBRES_LANDMARKS[PUNTOS[a]] or "left" in NOMBRES_LANDMARKS[PUNTOS[b]]
            color = (214, 120, 42) if izq else (52, 104, 235)   # BGR: azul izq., naranja der.
            cv2.line(im, px[a], px[b], color, 2, cv2.LINE_AA)
    for p in px:
        if p:
            cv2.circle(im, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.imwrite(destino, im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return True


def recorte(cap, frame, xy, destino, alto=320, margen=0.35, espejo=False):
    """JPG del frame recortado alrededor del esqueleto (con margen para la
    raqueta) y dibujado encima: asi un profesional lejos de la camara y alguien
    a dos metros se ven al mismo tamano."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    if not ok:
        return False
    pts = np.array([xy[NOMBRES_LANDMARKS[p]][frame] for p in PUNTOS])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 6:
        return False
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    lado = max(x1 - x0, y1 - y0) * (1 + 2 * margen)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = im.shape[:2]
    a, b = int(max(cx - lado * 0.4, 0)), int(min(cx + lado * 0.4, w))
    c, d = int(max(cy - lado / 2, 0)), int(min(cy + lado / 2, h))
    trozo = im[c:d, a:b].copy()
    k = alto / max(1, trozo.shape[0])
    trozo = cv2.resize(trozo, (max(1, int(trozo.shape[1] * k)), alto))
    for i, j in HUESOS:
        p, q = xy[NOMBRES_LANDMARKS[PUNTOS[i]]][frame], xy[NOMBRES_LANDMARKS[PUNTOS[j]]][frame]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            continue
        izq = "left" in NOMBRES_LANDMARKS[PUNTOS[i]] or "left" in NOMBRES_LANDMARKS[PUNTOS[j]]
        cv2.line(trozo, (int((p[0] - a) * k), int((p[1] - c) * k)),
                 (int((q[0] - a) * k), int((q[1] - c) * k)),
                 (214, 120, 42) if izq else (52, 104, 235), 2, cv2.LINE_AA)
    if espejo:
        trozo = cv2.flip(trozo, 1)
    cv2.imwrite(destino, trozo, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def representativo(golpes, clave="giro_hombros"):
    """Indice del golpe mas cercano a la mediana de `clave` (el mas tipico de
    la sesion), o el primero si no hay datos. Funcion pura."""
    v = np.array([np.nan if g.get(clave) is None else g[clave] for g in golpes], float)
    if not len(v):
        return None
    if not np.isfinite(v).any():
        return 0
    return int(np.nanargmin(np.abs(v - np.nanmedian(v))))


def fases_de(cap, xy, s, t, prefijo, destino, espejo=False):
    """Recortes de maxima torsion, impacto y final del giro de un golpe."""
    prep, fin = mg.momentos(s, t)
    fotos = {}
    for fase, f in (("preparacion", prep), ("impacto", t), ("final", fin)):
        if f is not None and recorte(cap, int(f), xy, os.path.join(destino, f"{prefijo}_{fase}.jpg"),
                                     espejo=espejo):
            fotos[fase] = {"frame": int(f), "src": f"{prefijo}_{fase}.jpg"}
    return fotos


def comparativa(clip, video, procesado, destino, procesado_ref, videos_ref):
    """Tu golpe tipico frente al golpe tipico de cada profesional con el mismo
    golpe y vista, fase a fase y a la misma escala."""
    ruta_ref = os.path.join(RAIZ, "data", "processed", "metricas_referencia.csv")
    if not clip["golpes"] or not os.path.isfile(ruta_ref):
        return []
    ref = ac.cargar_referencias(ruta_ref)
    ref = ref[(ref["golpe"] == clip["golpe"]) & (ref["vista"] == clip["vista"])]
    filas = []
    fps, ancho, alto = mg.info_video(video)
    _, xy, vis = cargar(os.path.join(procesado, f"{clip['nombre']}_corregido.csv"), ancho, alto)
    s = mg.series(xy, "right")
    i = representativo(clip["golpes"])
    g = clip["golpes"][i]
    sentido = mg.direccion_golpe(s, int(g["impacto_frame"]))
    cap = cv2.VideoCapture(video)
    filas.append({"quien": "Tú", "golpe": g["n"], "giro_hombros": g.get("giro_hombros"),
                  "fotos": fases_de(cap, xy, s, int(g["impacto_frame"]), "comp_tu", destino)})
    cap.release()
    for jugador, x in ref.groupby("jugador"):
        x = x.reset_index(drop=True)
        j = representativo(x.to_dict("records"))
        fila = x.iloc[j]
        v = os.path.join(videos_ref, f"{fila['clip']}.mp4")
        if not os.path.isfile(v):
            continue
        fps_r, ancho_r, alto_r = mg.info_video(v)
        _, xy_r, _ = cargar(os.path.join(procesado_ref, f"{fila['clip']}_corregido.csv"), ancho_r, alto_r)
        s_r = mg.series(xy_r, "right")
        # si golpea hacia el otro lado de la imagen, se muestra en espejo
        espejo = bool(np.isfinite(sentido) and mg.direccion_golpe(s_r, int(fila["impacto"])) == -sentido)
        cap = cv2.VideoCapture(v)
        prefijo = "comp_" + "".join(ch for ch in jugador.lower() if ch.isalnum())
        filas.append({"quien": jugador, "clip": fila["clip"], "giro_hombros": fila.get("giro_hombros"),
                      "espejo": espejo,
                      "fotos": fases_de(cap, xy_r, s_r, int(fila["impacto"]), prefijo, destino, espejo)})
        cap.release()
    return limpiar(filas)


def datos_clip(nombre, video, procesado, golpe, vista, golpes, sesion, extra):
    """Todo lo que pinta la plantilla, mas los ficheros de la carpeta del clip.

    golpes: lista de dicts con 'impacto_frame' y las metricas de ese golpe.
    sesion: filas de analizar_clip.comparar, o None si no hay comparacion.
    """
    pre = os.path.join(procesado, nombre)
    fps, ancho, alto = mg.info_video(video)
    n, xy, vis = cargar(f"{pre}_corregido.csv", ancho, alto)
    s = mg.series(xy, "right")
    s["vis_muneca"] = np.nan_to_num(vis["right_wrist"], nan=0.0)

    corr = leer_correcciones(f"{pre}_corregido.csv")
    volteos, picos = eventos_de_correccion(corr)
    tramos_raqueta, cajas = raqueta(f"{pre}_raqueta.csv", n)
    cortes = []
    if os.path.isfile(f"{pre}_cortes.csv"):
        cortes = [int(f) for f in pd.read_csv(f"{pre}_cortes.csv")["frame"]]

    detectado = np.isfinite(vis["nose"]) | np.isfinite(vis["left_hip"])
    brazo = np.nanmin(np.vstack([vis[f"right_{p}"] for p in ("shoulder", "elbow", "wrist")]), 0)
    piernas = np.nanmedian(np.vstack([vis[p] for p in mg.PUNTOS_PIERNAS]), 0)

    fichas = []
    for i, g in enumerate(golpes):
        t = int(g["impacto_frame"])
        prep, fin = mg.momentos(s, t)
        fichas.append({**g, "n": i + 1, "preparacion": prep, "final": fin})

    clip = {
        "nombre": nombre, "golpe": golpe, "vista": vista, "fps": round(fps, 2),
        "ancho": ancho, "alto": alto, "frames": n, "duracion": round(n / fps, 2),
        "deteccion_pct": round(100 * detectado.mean(), 1),
        "visible_pct": {
            "brazo dominante": round(100 * float(np.mean(np.nan_to_num(brazo) >= mg.UMBRAL_VIS)), 1),
            "piernas": round(100 * float(np.mean(np.nan_to_num(piernas) >= mg.UMBRAL_VIS)), 1),
        },
        "linea": {"brazo": reducir(brazo), "piernas": reducir(piernas)},
        "sin_deteccion": tramos(~detectado),
        "volteos": volteos, "picos": picos, "cortes": cortes, "raqueta": tramos_raqueta,
        "golpes": fichas, "sesion": sesion, "con_golpes": True,
        "puntos": [NOMBRES_LANDMARKS[p] for p in PUNTOS], "huesos": HUESOS,
        "escala": ESCALA, **extra,
    }
    clip["avisos"] = avisos(clip)
    return clip, xy, cajas, corr


def escribir_clip(clip, video, procesado, cajas, corr, sin_video=False, ref_dirs=None):
    """Carpeta data/revision/<nombre>/ con index.html, datos.js, video y miniaturas."""
    pre = os.path.join(procesado, clip["nombre"])
    destino = os.path.join(DIR_REVISION, clip["nombre"])
    os.makedirs(destino, exist_ok=True)

    puntos = esqueleto(f"{pre}_corregido.csv", clip["frames"])
    originales = originales_distintos(f"{pre}_landmarks.csv", corr)
    marcadas = {}
    for f, lid in corr[["frame", "landmark_id"]].astype(int).itertuples(index=False):
        if lid in PUNTOS:
            marcadas.setdefault(f, []).append(PUNTOS.index(lid))

    cap = cv2.VideoCapture(video)
    for g in clip["golpes"]:
        g["miniaturas"] = {}
        for clave in ("preparacion", "impacto_frame", "final"):
            f = g.get(clave)
            if f is None:
                continue
            fichero = f"golpe{g['n']}_{clave.split('_')[0]}.jpg"
            if miniatura(cap, int(f), puntos, os.path.join(destino, fichero)):
                g["miniaturas"][clave.split("_")[0]] = {"frame": int(f), "src": fichero}
    medio = clip["golpes"][0]["impacto_frame"] if clip["golpes"] else clip["frames"] // 2
    miniatura(cap, int(medio), puntos, os.path.join(destino, "portada.jpg"), alto=240)
    cap.release()

    if not sin_video or not os.path.isfile(os.path.join(destino, "video.mp4")):
        recodificar(video, os.path.join(destino, "video.mp4"))

    if ref_dirs and clip.get("grupo") == "propio":
        clip["comparativa"] = comparativa(clip, video, procesado, destino, *ref_dirs)

    datos = {"clip": clip, "esqueleto": puntos, "originales": originales,
             "corregidas": marcadas, "cajas": cajas}
    with open(os.path.join(destino, "datos.js"), "w") as f:
        f.write("window.DATOS = ")
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                  default=lambda o: None)
        f.write(";\n")
    shutil.copy(PLANTILLA, os.path.join(destino, "index.html"))
    with open(os.path.join(destino, "resumen.json"), "w") as f:
        json.dump(resumen(clip), f, ensure_ascii=False, indent=1)
    return destino


def resumen(clip):
    """Lo que necesita la pagina indice de cada clip. Funcion pura."""
    sesion = clip.get("sesion") or []
    return {
        "nombre": clip["nombre"], "titulo": clip.get("titulo", clip["nombre"]),
        "grupo": clip.get("grupo", "propio"), "golpe": clip["golpe"], "vista": clip["vista"],
        "fps": clip["fps"], "duracion": clip["duracion"], "golpes": len(clip["golpes"]),
        "deteccion_pct": clip["deteccion_pct"],
        "volteos": sum(b - a + 1 for a, b in clip["volteos"]), "picos": len(clip["picos"]),
        "a_revisar": [m["metrica"] for m in sesion if m.get("valoracion") == "a revisar"],
        "comparadas": sum(1 for m in sesion if m.get("valoracion")),
        "avisos": len(clip["avisos"]),
    }


def limpiar(v):
    """NaN/inf -> None, recursivo, para JSON valido. Funcion pura."""
    if isinstance(v, dict):
        return {k: limpiar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [limpiar(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def desde_informe(ruta_informe):
    """Clip propio a partir del _informe.json de analizar_clip.py."""
    with open(ruta_informe) as f:
        inf = json.load(f)
    procesado = os.path.dirname(ruta_informe)
    nombre = os.path.basename(ruta_informe)[:-len("_informe.json")]
    golpes = [{k: v for k, v in g.items() if k != "impacto"} for g in inf["golpes"]]
    extra = {"titulo": nombre, "grupo": "propio", "impactos": inf.get("impactos", "automaticos")}
    clip, _, cajas, corr = datos_clip(nombre, inf["video"], procesado, inf["golpe"], inf["vista"],
                                      golpes, inf["sesion"], extra)
    return limpiar(clip), inf["video"], procesado, cajas, corr


def referencias(procesado, videos):
    """Un clip por video de references/manifest.csv. Los golpes anotados se
    miden y cada profesional se compara con los demas (como en
    validar_comparacion), para ver en que le marcaria el informe."""
    manifest = pd.read_csv(os.path.join(RAIZ, "references", "manifest.csv"))
    anot = pd.concat([pd.read_csv(os.path.join(RAIZ, "references", f))
                      for f in ("golpes.csv", "golpes_auto.csv")], ignore_index=True)
    ref = ac.cargar_referencias(os.path.join(RAIZ, "data", "processed", "metricas_referencia.csv"))
    for fila in manifest.itertuples(index=False):
        video = os.path.join(videos, f"{fila.nombre}.mp4")
        if not (os.path.isfile(video)
                and os.path.isfile(os.path.join(procesado, f"{fila.nombre}_corregido.csv"))):
            print(f"  {fila.nombre}: sin video o sin procesar, se salta")
            continue
        suyos = anot[anot["clip"] == fila.nombre]
        # golpe y vista dominantes del clip (algunos mezclan derecha y reves)
        clave = (suyos.groupby(["golpe", "vista"]).size().idxmax() if len(suyos)
                 else (fila.golpe, fila.vista))
        golpe, vista = clave
        sel = suyos[(suyos["golpe"] == golpe) & (suyos["vista"] == vista)]
        fps, ancho, alto = mg.info_video(video)
        medidas = mg.medir_golpes(os.path.join(procesado, f"{fila.nombre}_corregido.csv"),
                                  ancho, alto, fps, zip(sel["impacto"], sel["golpe"]))
        golpes = [{"impacto_frame": int(m["impacto"]), "impacto_s": round(m["impacto"] / fps, 2),
                   **{k: v for k, v in m.items() if k != "impacto"}} for m in medidas]
        sesion = None
        if golpes:
            tabla = pd.DataFrame(medidas)
            medidos = {c: int(tabla[c].notna().sum()) for c in ac.FIABILIDAD if c in tabla}
            valores = {c: float(tabla[c].median()) if medidos[c] >= ac.MIN_GOLPES_SESION
                       else np.nan for c in medidos}
            otros = ref[ref["jugador"] != fila.jugador]
            sesion = ac.marcar_pocos_golpes(ac.comparar(valores, otros, golpe, vista, fps),
                                            medidos, len(golpes))
        extra = {"titulo": f"{fila.jugador} · {fila.nombre}", "grupo": "referencia",
                 "jugador": fila.jugador, "impactos": "anotados" if golpes else "sin anotar",
                 "nota_comparacion": f"{fila.jugador} comparado con los demas profesionales "
                                     "(el queda fuera del rango de referencia)"}
        clip, _, cajas, corr = datos_clip(fila.nombre, video, procesado, golpe, vista, golpes,
                                          sesion, extra)
        clip["con_golpes"] = bool(golpes)
        clip["avisos"] = avisos(clip)
        if not golpes:
            clip["avisos"].append("Clip sin golpes anotados: solo sirve para revisar el esqueleto.")
        yield limpiar(clip), video, procesado, cajas, corr


def escribir_indice():
    """data/revision/index.html con todos los clips revisados."""
    filas = []
    for d in sorted(os.listdir(DIR_REVISION)):
        ruta = os.path.join(DIR_REVISION, d, "resumen.json")
        if os.path.isfile(ruta):
            with open(ruta) as f:
                filas.append(json.load(f))
    filas.sort(key=lambda r: (r["grupo"] != "propio", r["titulo"]))
    with open(os.path.join(os.path.dirname(PLANTILLA), "indice.html")) as f:
        plantilla = f.read()
    datos = json.dumps(filas, ensure_ascii=False).replace("</", "<\\/")
    with open(os.path.join(DIR_REVISION, "index.html"), "w") as f:
        f.write(plantilla.replace("/*DATOS*/[]", datos))
    return len(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("informes", nargs="*", help="<nombre>_informe.json de analizar_clip.py")
    parser.add_argument("--referencias", action="store_true",
                        help="Revisa tambien los clips de references/manifest.csv")
    parser.add_argument("--procesado-ref", default=os.path.join(RAIZ, "data", "processed", "ref_v2"))
    parser.add_argument("--videos-ref", default=os.path.join(RAIZ, "data", "raw", "ref"))
    parser.add_argument("--sin-video", action="store_true",
                        help="No vuelve a recodificar el video si ya esta (mas rapido)")
    args = parser.parse_args()
    if not args.informes and not args.referencias:
        parser.error("indica algun _informe.json o --referencias")

    trabajos = [desde_informe(r) for r in args.informes]
    ref_dirs = (args.procesado_ref, args.videos_ref)
    for clip, video, procesado, cajas, corr in trabajos:
        print(f"  {clip['nombre']} -> "
              f"{escribir_clip(clip, video, procesado, cajas, corr, args.sin_video, ref_dirs)}")
    if args.referencias:
        for clip, video, procesado, cajas, corr in referencias(args.procesado_ref, args.videos_ref):
            print(f"  {clip['nombre']} -> {escribir_clip(clip, video, procesado, cajas, corr, args.sin_video)}")
    n = escribir_indice()
    print(f"\nIndice con {n} clips -> {html.escape(os.path.join(DIR_REVISION, 'index.html'))}")


if __name__ == "__main__":
    main()
