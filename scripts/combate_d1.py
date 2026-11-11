"""Revisión opcional de encuadre con D1 Ultra sobre fotogramas de combate.

Envía imágenes a Liquid AI; no interpreta movimientos ni cambia los diagnósticos.
"""
import base64
import argparse
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
import tempfile

URL = "https://api.liquid.ai/decisions/v1/systemone"
MODELO = "d1-ultra"
PREGUNTAS = {
    "cuerpo_entero": "Is one person visible from head to both feet without being cut off by the image edge?",
    "varias_personas": "Are two or more people visible in this image?",
    "vista_frontal": "Is the main person facing the camera, rather than seen from the side or behind?",
}
ETIQUETAS = {"cuerpo_entero": "Cuerpo entero visible", "varias_personas": "Varias personas",
             "vista_frontal": "Persona de frente"}


class ErrorD1(Exception):
    """Error de configuración, vídeo o servicio D1, sin credenciales en el mensaje."""


def huella(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def extraer_fotogramas(video, cantidad=5):
    """Muestreo uniforme en el interior del clip, sin guardar imágenes en disco."""
    import cv2
    if not 1 <= cantidad <= 10:
        raise ErrorD1("El número de fotogramas debe estar entre 1 y 10.")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ErrorD1("No se pudo abrir el vídeo.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total < 1 or not math.isfinite(fps) or fps <= 0:
            raise ErrorD1("No se pudo determinar la duración o los fps del vídeo.")
        frames = []
        for i in range(cantidad):
            frame = min(total - 1, int((i + 0.5) * total / cantidad))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, imagen = cap.read()
            if not ok:
                raise ErrorD1(f"No se pudo leer el fotograma {frame}.")
            h, w = imagen.shape[:2]
            if max(h, w) > 768:
                escala = 768 / max(h, w)
                imagen = cv2.resize(imagen, (max(1, round(w * escala)), max(1, round(h * escala))),
                                    interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ErrorD1(f"No se pudo codificar el fotograma {frame}.")
            frames.append((frame, frame / fps, jpeg.tobytes()))
        return frames
    finally:
        cap.release()


def consultar(imagen, clave, abrir=urllib.request.urlopen):
    carga = {"model": MODELO, "images": ["data:image/jpeg;base64," + base64.b64encode(imagen).decode("ascii")],
             "state": "A single frame sampled from a combat training video. Judge only what is visible in this image.",
             "questions": {k: {"type": "noul", "instructions": v} for k, v in PREGUNTAS.items()}}
    req = urllib.request.Request(URL, data=json.dumps(carga).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + clave, "Content-Type": "application/json"})
    try:
        with abrir(req, timeout=45) as respuesta:
            datos = json.load(respuesta)
    except urllib.error.HTTPError as e:
        raise ErrorD1(f"Liquid AI respondió con HTTP {e.code}; comprueba clave, acceso y saldo.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ErrorD1(f"No se pudo contactar con Liquid AI ({type(e).__name__}).") from None
    except (ValueError, UnicodeError):
        raise ErrorD1("Liquid AI devolvió una respuesta que no es JSON válido.") from None
    try:
        probabilidades = {k: datos["answers"][k]["noul"] for k in PREGUNTAS}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or
               not 0 <= v <= 1 for v in probabilidades.values()):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ErrorD1("Liquid AI devolvió probabilidades incompletas o inválidas.") from None
    return probabilidades


def revisar(video, cantidad=5, clave=None):
    clave = clave or os.environ.get("LIQUID_API_KEY")
    if not clave:
        raise ErrorD1("Falta LIQUID_API_KEY; configura la clave de Liquid AI para usar --d1.")
    frames = extraer_fotogramas(video, cantidad)
    resultados = []
    for numero, segundos, imagen in frames:
        resultados.append({"frame": numero, "segundos": round(segundos, 3),
                           "probabilidades": consultar(imagen, clave)})
    return {"modelo": MODELO, "video_sha256": huella(video), "fotogramas": resultados}


def guardar(resultado, ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    fd, temporal = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(ruta)), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)


def main():
    parser = argparse.ArgumentParser(description="Revisar fotogramas con D1 Ultra de Liquid AI")
    parser.add_argument("video")
    parser.add_argument("--salida", required=True, help="JSON de salida; usa <clip>_d1.json junto a los datos")
    parser.add_argument("--fotogramas", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()
    try:
        resultado = revisar(args.video, args.fotogramas)
        guardar(resultado, args.salida)
    except (ErrorD1, OSError) as e:
        parser.exit(1, f"D1: {e}\n")
    print(f"Revisión D1 de {len(resultado['fotogramas'])} fotogramas -> {args.salida}")


if __name__ == "__main__":
    main()
