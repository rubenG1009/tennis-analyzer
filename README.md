# tennis-analyzer

Análisis de técnica de golpes de tenis a partir de vídeo.

## Qué es

El pipeline previsto, de principio a fin:

```
vídeo -> pose estimation -> serie temporal de landmarks
      -> segmentación de repeticiones -> métricas biomecánicas
      -> agente LLM -> informe
```

La idea es partir de un clip de un jugador golpeando, reducirlo a una serie
temporal de articulaciones, cortar esa serie en golpes individuales, calcular
métricas sobre cada golpe y que un LLM redacte un informe técnico a partir de
esos números.

## Estado

**Implementadas las cuatro etapas.**

| Etapa | Estado |
|---|---|
| 1. Vídeo -> serie temporal de landmarks | implementada, con corrección de inversiones y evaluación de calidad |
| 2. Segmentación de repeticiones | implementada (`segmentar_golpes.py`) |
| 3. Métricas biomecánicas | implementada (`metricas_golpe.py`, `analizar_clip.py`) |
| 4. Informe redactado por un LLM | implementada (`redactar_informe.py`) |

No hay backend, ni API, ni contenedores, y no está previsto añadirlos por ahora.

## Instalación

Requiere Python 3.9 o superior y macOS o Linux.

```bash
git clone https://github.com/rubenG1009/tennis-analyzer.git
cd tennis-analyzer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Los modelos no están en el repositorio: hay que bajarlos.
./scripts/download_models.sh
```

`download_models.sh` descarga `pose_landmarker_heavy.task` (~30 MB, el que usa
el pipeline) y `pose_landmarker_lite.task` (~6 MB, el que usan los tests) a
`models/`. Es idempotente: si ya están, no hace nada.

### Nota sobre la versión de mediapipe

`requirements.txt` clava **mediapipe 1.0.0** a propósito. La 1.0.1 aborta el
proceso al crear el `PoseLandmarker` en macOS arm64:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
```

El detalle está en [docs/decisiones.md](docs/decisiones.md). No actualices
mediapipe sin comprobar antes que el pipeline sigue arrancando.

Si `pip install` falla con `TypeError: encode() argument 'encoding' must be str,
not None`, es un bug de `compileall` en Python 3.9: reinténtalo con
`pip install --no-compile -r requirements.txt`.

## Uso

```bash
python scripts/extract_pose.py data/raw/mi_clip.mp4
```

Opciones:

| Argumento | Por defecto | Qué hace |
|---|---|---|
| `video` (posicional) | — | Vídeo de entrada |
| `--model` | `models/pose_landmarker_heavy.task` | Fichero `.task` del landmarker |
| `--out-dir` | `data/processed` | Dónde escribir las salidas |

Genera dos ficheros, con el nombre derivado del vídeo de entrada:

- `data/processed/<nombre>_landmarks.csv` — la serie temporal.
- `data/processed/<nombre>_debug.mp4` — el vídeo con el esqueleto superpuesto,
  para comprobar a ojo que la detección es razonable. Las articulaciones se
  pintan verdes si su `visibility` alcanza el umbral (0.5) y rojas si no.

Por consola imprime un informe con la resolución, los fps, los frames leídos
frente a los declarados, el porcentaje de frames sin detección y, para los
landmarks del lado derecho, la `visibility` media, la mínima y el porcentaje de
frames que alcanzan el umbral.

## Clips de referencia y calidad del esqueleto

```bash
# 1. Bajar los clips de referencia de references/manifest.csv a data/raw/ref/
#    (necesita uv y node; ver docs/decisiones.md)
python scripts/download_references.py

# 2. Extraer con deteccion de raqueta
python scripts/extract_pose.py data/raw/ref/sv_roddick.mp4 \
    --raqueta models/efficientdet_lite2.tflite --out-dir data/processed/ref

# 3. Corregir inversiones izquierda/derecha y picos -> <nombre>_corregido.csv
python scripts/corregir_esqueleto.py data/raw/ref/sv_roddick.mp4 \
    data/processed/ref/sv_roddick_landmarks.csv

# 4. Medir sintomas de error (pares video csv, tantos como se quiera)
python scripts/evaluar_esqueleto.py \
    data/raw/ref/sv_roddick.mp4 data/processed/ref/sv_roddick_landmarks.csv \
    data/raw/ref/sv_roddick.mp4 data/processed/ref/sv_roddick_corregido.csv
```

Que mide el evaluador, que errores se encontraron y cuanto los reduce el
corrector esta en [docs/hallazgos.md](docs/hallazgos.md).

## Analizar un clip propio

```bash
python scripts/analizar_clip.py data/raw/mi_derecha.mp4 --golpe derecha --vista lateral
```

Hace todo el pipeline (landmarks, corrección, golpes, métricas) y compara la
sesión con los profesionales de referencia. Antes hay que generar las
referencias una vez:

```bash
python scripts/download_references.py
# extraer + corregir cada clip de data/raw/ref/ en data/processed/ref_v2/ (ver arriba)
python scripts/metricas_golpe.py referencias
```

Cómo grabar para que sirva:

- **Un tipo de golpe por clip** (derecha o revés o saque): no se clasifican solos.
- **Cámara lenta del móvil (120–240 fps)**: a 30 fps el brazo sale movido en el
  impacto y las métricas del brazo no se comparan.
- **Cámara a la altura de la cadera en el lateral de la pista**, como en los
  clips de referencia, y el jugador entero en plano.
- Solo diestros por ahora.

Qué se compara y con qué fiabilidad está en [docs/hallazgos.md](docs/hallazgos.md).

## Informe en lenguaje natural

```bash
export ANTHROPIC_API_KEY=...
uv run scripts/redactar_informe.py data/processed/mi_derecha_informe.json
```

Convierte el `_informe.json` de `analizar_clip.py` en un informe de entrenador
(`_informe.md`) con Claude. Se ejecuta con `uv run` porque el SDK de Anthropic
exige Python ≥ 3.10 y el resto del proyecto está en 3.9; `uv` instala lo que
haga falta la primera vez. Con `--dry-run` imprime el prompt sin llamar a la
API ni necesitar clave.

El modelo no ve el vídeo: solo la tabla de la sesión, con el estado y la
fiabilidad de cada métrica. El prompt le prohíbe valorar las métricas que no se
pudieron comparar y presentar "fuera de rango" como un error. Después, el
script busca en el texto cifras que no salgan del informe y las avisa por
consola.

## Tests

```bash
python -m pytest -v
```

Los tests cubren las constantes hardcodeadas de BlazePose, la monotonía de los
timestamps, el contrato estructural del CSV, la función de estadísticas y el
manejo de errores del CLI. No testean mediapipe ni OpenCV: son dependencias de
terceros.

Los tres tests de integración usan el modelo **lite** en vez del heavy, para
que la suite tarde segundos en lugar de minutos; lo que verifican es la forma
del CSV, que no depende de qué modelo lo haya rellenado. Si `models/pose_landmarker_lite.task` no está, esos tres tests **se saltan** (`skipped`) en
vez de fallar. Ejecuta `./scripts/download_models.sh` para que corran.

## Formato de salida

El CSV está en **formato largo**: una fila por cada landmark y cada frame. Un
vídeo de N frames produce N × 33 filas.

| Columna | Significado |
|---|---|
| `frame` | Índice del frame, empezando en 0. Sin huecos. |
| `t_ms` | Milisegundos desde el inicio, `round(frame * 1000 / fps)`. Estrictamente creciente. |
| `landmark_id` | Índice del landmark en BlazePose, de 0 a 32. |
| `landmark` | Nombre del landmark (`right_wrist`, `left_knee`...). |
| `x` | Coordenada horizontal, **normalizada 0–1** respecto al ancho de la imagen. |
| `y` | Coordenada vertical, **normalizada 0–1** respecto al alto de la imagen. |
| `z` | Profundidad relativa estimada. **No usar**, ver limitaciones. |
| `visibility` | Confianza de 0 a 1 en que el landmark sea visible y esté bien situado. |

`x` e `y` son proporciones del tamaño de la imagen, no píxeles: para pasar a
píxeles hay que multiplicar por el ancho y el alto. Esto hace que la serie sea
comparable entre clips de distinta resolución, pero **no** entre clips con
distinto encuadre.

Los frames en los que no se detecta ninguna pose **se escriben igualmente**, con
sus 33 filas y las celdas `x`, `y`, `z` y `visibility` vacías. El eje temporal
nunca tiene huecos silenciosos: un frame que falta es un frame que no se leyó,
no un frame en el que no se detectó nada.

## Limitaciones conocidas

**`z` no se usa, y no debería usarse.** MediaPipe la estima por regresión a
partir de una sola vista 2D: no es una medida de profundidad, es una conjetura
del modelo sobre cómo se proyectaría el cuerpo en 3D. No tiene escala métrica,
depende del encuadre y es inestable entre frames contiguos. Se vuelca al CSV por
completitud, pero cualquier métrica construida sobre ella sería ruido con
aspecto de señal.

**`world_landmarks` no se extraen**, por el mismo motivo. MediaPipe los ofrece en
metros aproximados y con el origen en la cadera, lo que parece justo lo que hace
falta para biomecánica, pero salen de la misma regresión monocular. Una medida en
metros que viene de adivinar la profundidad desde un único punto de vista sigue
siendo una adivinanza, solo que con unidades. Para ángulos 3D fiables harían
falta dos cámaras calibradas.

**La calidad del clip manda.** `visibility` cae mucho en articulaciones ocluidas
o lejanas. En el clip de prueba, la muñeca derecha solo alcanza el umbral en el
35 % de los frames, lo que la inutiliza para métricas de golpeo. Ver
[docs/hallazgos.md](docs/hallazgos.md).

## Documentación

- [docs/decisiones.md](docs/decisiones.md) — decisiones técnicas y por qué.
- [docs/hallazgos.md](docs/hallazgos.md) — lo que se ha observado en clips reales.
