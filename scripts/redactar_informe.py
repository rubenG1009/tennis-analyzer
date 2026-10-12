#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=1.0"]
# ///
"""Etapa 4: redacta con un LLM el informe tecnico de un clip a partir de las
metricas que escribe analizar_clip.py.

    uv run scripts/redactar_informe.py data/processed/mi_derecha_informe.json

Se ejecuta con `uv run` y no desde el venv porque anthropic 1.x exige
Python >= 3.10 y el proyecto sigue en 3.9 por mediapipe (mismo arreglo que
yt-dlp, ver docs/decisiones.md). Este script solo lee el JSON: no necesita
mediapipe ni OpenCV.

El LLM no ve el video ni los landmarks, solo la tabla de la sesion. Su trabajo
es explicar los numeros en lenguaje de entrenador, no medir nada. Para que no
se invente datos:

- El prompt le da cada metrica con su estado y su fiabilidad tal como los
  calculo analizar_clip.py, y le prohibe comentar las que no se compararon.
- Despues de redactar, `numeros_no_respaldados` busca en el texto cifras que no
  salgan del informe y las avisa por consola. No corrige nada: avisa.

Con --dry-run imprime el prompt sin llamar a la API (no hace falta clave).
Escribe <nombre>_informe.md junto al JSON.
"""

import argparse
import json
import os
import re
import sys

MODELO = "claude-opus-5-5"
MAX_TOKENS = 16000

# Cifras que el prompt le da al modelo aunque no esten en el JSON (consejos de
# grabacion), para que el verificador no las marque.
CIFRAS_DEL_PROMPT = {30, 100, 120, 240, 180}

SISTEMA = """\
Eres un entrenador de tenis que explica a un jugador aficionado el analisis \
automatico de un clip suyo. Escribes en espanol, de tu a tu, claro y sin jerga \
innecesaria.

El analisis lo ha hecho un programa de vision por computador: estima el \
esqueleto en 2D con una sola camara, detecta cada golpe, mide unas pocas \
metricas por golpe y compara la mediana de la sesion con el rango de varios \
profesionales grabados con la misma vista. Tu no ves el video: solo tienes la \
tabla que te paso.

Reglas, por orden de importancia:

1. Usa solo los numeros de la tabla. No calcules metricas nuevas ni supongas \
nada que la tabla no diga. Si citas un valor, redondealo como mucho a un \
decimal.
2. "fuera del rango" no significa "mal": significa que ningun profesional de \
referencia lo hace asi de forma habitual. Presentalo como algo que merece \
mirarse, nunca como un error seguro.
3. "dentro del rango" tampoco significa "perfecto": significa que es compatible \
con lo que hacen los profesionales de referencia.
4. Respeta la fiabilidad: las metricas de fiabilidad "media" mencionalas con \
cautela. No comentes el valor de las que tengan estado "no comparable", "sin \
dato" o "sin referencias suficientes": di solo por que no se han podido \
valorar y que haria falta (por ejemplo, grabar en camara lenta de 120-240 fps).
5. Si hay pocos golpes (menos de 5), avisa de que la sesion es corta y las \
conclusiones son provisionales.
6. No des consejos de tecnica que no se apoyen en una metrica de la tabla.

Formato, en Markdown:

## Resumen
Dos o tres frases con lo mas importante.

## Lo que ya haces como los profesionales
## Lo que merece mirarse
(si no hay nada fuera de rango, dilo en una frase)

## Lo que este analisis no ha podido valorar
## Para el proximo clip
Uno a tres consejos de grabacion, solo si ayudarian a valorar mas cosas.\
"""


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def tabla_sesion(informe):
    """Tabla Markdown con una fila por metrica de la sesion. Funcion pura."""
    filas = ["| metrica | que mide | unidad | valor | rango pro | estado | fiabilidad |",
             "|---|---|---|---|---|---|---|"]
    for m in informe["sesion"]:
        rango = (f"{_fmt(m['min'])} a {_fmt(m['max'])} "
                 f"({m['jugadores']} jugadores, {m['golpes']} golpes)") if "min" in m else "-"
        filas.append(f"| {m['metrica']} | {m['descripcion']} | {m['unidad']} | "
                     f"{_fmt(m['valor'])} | {rango} | {m['estado']} | {m['fiabilidad']} |")
    return "\n".join(filas)


def construir_mensaje(informe):
    """Mensaje de usuario con el contexto del clip y la tabla. Funcion pura."""
    golpe = informe["golpe"].replace("_", " ")
    return (f"Clip: {golpe}, vista {informe['vista']}, grabado a {informe['fps']:.0f} fps.\n"
            f"Golpes de {golpe} detectados: {informe['golpes_detectados']} "
            f"(valores de la sesion = mediana de esos golpes).\n\n"
            f"{tabla_sesion(informe)}\n\n"
            "Redacta el informe.")


def numeros_no_respaldados(texto, informe):
    """Cifras del texto que no salen del informe ni del prompt. Funcion pura.

    Acepta coma o punto decimal y cualquier redondeo de un valor numerico del
    informe (valores, rangos, conteos, fps): una cifra con d decimales vale si
    esta a menos de media unidad de su ultimo decimal de algun valor. Se compara
    asi y no con round() porque round(92.05, 1) da 92.0 en coma flotante y una
    persona escribe 92,1. Ignora enteros menores que 10: numeran listas, frames
    o golpes y dan falsos avisos."""
    validos = set(CIFRAS_DEL_PROMPT)

    def recoger(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            validos.add(abs(x))
        elif isinstance(x, dict):
            for y in x.values():
                recoger(y)
        elif isinstance(x, list):
            for y in x:
                recoger(y)

    recoger(informe)
    sospechosos = []
    for crudo in re.findall(r"\d+(?:[.,]\d+)?", texto):
        n = float(crudo.replace(",", "."))
        if n.is_integer() and n < 10:
            continue
        decimales = len(re.split(r"[.,]", crudo)[1]) if re.search(r"[.,]", crudo) else 0
        tolerancia = 0.5 * 10 ** -decimales + 1e-9
        if not any(abs(n - v) <= tolerancia for v in validos):
            sospechosos.append(crudo)
    return sospechosos


def redactar(informe, cliente=None):
    """Llama al modelo y devuelve el texto del informe."""
    if cliente is None:
        import anthropic  # aqui y no arriba: los tests importan el modulo en 3.9
        cliente = anthropic.Anthropic()
    # fallbacks="default": si un clasificador de seguridad rechaza la peticion
    # (un falso positivo es posible), la API la repite en el modelo que
    # Anthropic recomienda para esa categoria en vez de devolver el rechazo.
    with cliente.beta.messages.stream(
        model=MODELO,
        max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium"},
        system=SISTEMA,
        messages=[{"role": "user", "content": construir_mensaje(informe)}],
    ) as stream:
        respuesta = stream.get_final_message()

    if respuesta.stop_reason == "refusal":
        raise RuntimeError(f"el modelo rechazo la peticion: {respuesta.stop_details}")
    if respuesta.stop_reason == "max_tokens":
        raise RuntimeError("el informe se corto por max_tokens")
    return "".join(b.text for b in respuesta.content if b.type == "text").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("informe", help="<nombre>_informe.json de analizar_clip.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime el prompt y sale, sin llamar a la API")
    args = parser.parse_args()

    with open(args.informe) as f:
        informe = json.load(f)

    if args.dry_run:
        print(SISTEMA, "\n\n---\n", construir_mensaje(informe), sep="")
        return
    if informe["golpes_detectados"] == 0:
        sys.exit("ERROR: el informe no tiene golpes; no hay nada que redactar")

    texto = redactar(informe)
    ruta = re.sub(r"_informe\.json$", "", args.informe) + "_informe.md"
    if ruta == args.informe:
        ruta = os.path.splitext(args.informe)[0] + ".md"
    with open(ruta, "w") as f:
        f.write(texto + "\n")

    print(texto)
    sospechosos = numeros_no_respaldados(texto, informe)
    if sospechosos:
        print(f"\nAVISO: cifras que no salen del informe: {', '.join(sospechosos)}",
              file=sys.stderr)
    print(f"\nInforme -> {ruta}")


if __name__ == "__main__":
    main()
