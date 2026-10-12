# Decisiones técnicas

Una entrada por decisión. Formato: contexto, decisión, consecuencia.

---

## 2026-09-14 — Clavar mediapipe a 1.0.0

**Contexto.** El entorno tenía mediapipe 1.0.1 instalado y verificado a nivel de
import. Al crear el primer `PoseLandmarker` el proceso abortaba entero:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
    @ mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

El wheel de 1.0.1 para macOS arm64 viene compilado con el path Metal activo.
`TensorsToDetectionsCalculator` intenta inicializar el helper Metal, el servicio
GPU no está registrado en el grafo, y abseil hace `CHECK`-fail. No es el fichero
del modelo ni el código de este repositorio: es el paquete.

**Qué se probó antes de concluirlo:**

| Prueba | Resultado |
|---|---|
| `delegate=BaseOptions.Delegate.CPU` | crash idéntico |
| `MEDIAPIPE_DISABLE_GPU=1` | crash idéntico |
| `RunningMode.IMAGE` en vez de `VIDEO` | crash idéntico |
| Modelos `lite`, `full` y `heavy` | crash idéntico en los tres |
| mediapipe 1.0.0 | carga y detecta, 30/30 frames |
| mediapipe 0.10.35 | carga y detecta, 30/30 frames |

**Decisión.** Bajar a 1.0.0 y clavarlo en `requirements.txt`. Se eligió 1.0.0
sobre 0.10.35 porque es el cambio mínimo: sigue siendo la API 1.x, y en ella
`mp.solutions` tampoco existe, así que el código no cambia ni una línea.

**Consecuencia.** El proyecto no puede actualizar mediapipe sin volver a
comprobar que el `PoseLandmarker` arranca. El comentario que lo explica está en
`requirements.txt`, para que un `pip install -U` dentro de unos meses no
reintroduzca el crash en silencio.

---

## 2026-09-14 — CSV en formato largo, escrito fila a fila con `csv`

**Contexto.** Cada frame produce 33 landmarks con 4 valores cada uno. Caben dos
formatos: ancho (una fila por frame, 132 columnas) o largo (una fila por landmark
y frame, 8 columnas).

**Decisión.** Formato largo, y escritura incremental con el módulo `csv` de la
librería estándar en lugar de acumular en un `DataFrame` de pandas.

**Consecuencia.** El formato largo es el que esperan las herramientas de series
temporales y permite filtrar por landmark sin tocar el resto; a cambio el fichero
es más grande y hay que pivotar para operaciones por frame. La escritura fila a
fila mantiene el uso de memoria constante: un clip de 12 s son 11 715 filas, pero
un partido entero son cientos de miles y no tiene sentido tenerlas todas en RAM
para volcarlas al final. pandas se usará en la etapa de análisis, no aquí.

---

## 2026-09-14 — Los frames sin detección se escriben con celdas vacías

**Contexto.** Cuando MediaPipe no encuentra ninguna pose en un frame, lo natural
sería no escribir nada para ese frame.

**Decisión.** Escribir igualmente las 33 filas, con `x`, `y`, `z` y `visibility`
vacías.

**Consecuencia.** El eje temporal no tiene huecos silenciosos. Un frame ausente
del CSV significa inequívocamente "no se leyó", nunca "no se detectó nada", que
son dos cosas muy distintas para la segmentación de repeticiones que viene
después: un hueco de 8 frames en mitad de un swing es información, no ruido a
ignorar. El coste es que quien lea el CSV tiene que tratar los nulos
explícitamente, que es justo lo que se pretende forzar. Hay un test que falla si
alguien "optimiza" esto omitiendo las filas.

---

## 2026-09-14 — `z` y `world_landmarks` se ignoran

**Contexto.** MediaPipe ofrece una `z` por landmark y un juego aparte de
`world_landmarks` en metros aproximados con origen en la cadera. Para un proyecto
de biomecánica parece justo lo que hace falta.

**Decisión.** Volcar `z` al CSV por completitud pero no construir nada sobre
ella, y no extraer `world_landmarks` en absoluto.

**Consecuencia.** Las métricas quedan limitadas al plano de imagen. Es
deliberado: ambas magnitudes salen de una regresión monocular, es decir, de una
conjetura del modelo sobre la profundidad a partir de una sola vista 2D. No
tienen escala métrica real, dependen del encuadre y son inestables entre frames
contiguos. Una medida en metros derivada de adivinar la profundidad sigue siendo
una adivinanza, solo que con unidades, y produciría ángulos 3D con apariencia de
rigor. Para profundidad fiable harían falta dos cámaras calibradas.

---

## 2026-09-14 — Guarda de monotonía en `t_ms`

**Contexto.** `detect_for_video` exige timestamps estrictamente crecientes y
falla si se repiten. El timestamp se calcula como `round(frame * 1000 / fps)`.

**Decisión.** `calcular_t_ms` aplica la fórmula y, si el resultado no supera al
timestamp anterior, lo fuerza a `anterior + 1`.

**Consecuencia.** La serie es siempre estrictamente creciente sea cual sea el
fps. El umbral real en el que la guarda entra es **por encima de 1000 fps**: a
1000 exactos `round(frame * 1000 / 1000) == frame`, que ya crece de uno en uno,
así que la guarda no llega a activarse. A partir de ahí el redondeo colapsa dos
frames en el mismo milisegundo y la guarda empieza a actuar. Con material real
(24–240 fps) nunca se dispara, y hay un test que verifica que en ese rango el
valor es exactamente el de la fórmula, sin corrección.

---

## 2026-09-14 — Los tests de integración usan el modelo lite

**Contexto.** El pipeline usa el modelo heavy. Procesar 20 frames sintéticos con
él cuesta unos segundos por test, y la suite se ejecuta muchas veces al día.

**Decisión.** Los tres tests de integración usan `pose_landmarker_lite.task`.
Llevan `skipif` y se saltan con motivo visible si el fichero no está.

**Consecuencia.** La suite entera tarda ~2 s en vez de ~70 s. Es sano porque lo
que verifican esos tests es el **contrato estructural** del CSV (número de filas,
columnas, orden, ausencia de huecos), y eso no depende de qué modelo lo haya
rellenado. La precisión del heavy no está bajo test, y no debería estarlo: sería
testear a un tercero.

---

## 2026-09-14 — Umbral de `visibility` unificado a `>=`

**Contexto.** La especificación inicial pedía pintar la articulación verde si
`visibility >= 0.5`, pero contar en el informe el porcentaje de frames
`> 0.5`. La asimetría estaba en el código y había un test que la fijaba.

**Decisión.** Unificar ambos a `>=`, leyendo los dos la misma constante
`UMBRAL_VISIBILITY`.

**Consecuencia.** Un landmark ya no puede salir verde en el vídeo de debug y a la
vez no contar en el porcentaje del informe, que era una discrepancia capaz de
hacer perder un rato a quien comparase ambas salidas. En la práctica los números
no se movieron: ningún `visibility` cae exactamente en 0.5, así que el informe
sobre el clip de prueba da los mismos porcentajes que antes. El test que fijaba
la asimetría ahora fija el criterio unificado, y hay otro que comprueba que los
dos sitios siguen leyendo la misma constante.

---

## 2026-10-07 — yt-dlp se ejecuta con `uvx`, no desde el venv

**Contexto.** La última versión de yt-dlp que funciona con Python 3.9 es
2025.10.14, y YouTube ya la rechaza (`The page needs to be reloaded`). Las
versiones actuales exigen Python ≥ 3.10 y un runtime de JS.

**Decisión.** `download_references.py` llama a `uvx --python 3.12 --from
"yt-dlp[default]" yt-dlp --js-runtimes node`. ffmpeg viene de `imageio-ffmpeg`,
un binario estático instalable con pip.

**Consecuencia.** El proyecto sigue en 3.9 sin tocar mediapipe, y no hay que
instalar nada en el sistema salvo `uv` y `node`. Se descarga solo vídeo (sin
audio) en H.264, así OpenCV lo lee sin sorpresas.

---

## 2026-10-07 — Recrear el `PoseLandmarker` en cada corte de escena

**Contexto.** En modo VIDEO el modelo sigue la región del frame anterior. Tras
un corte se queda enganchado a quien esté en esa zona (ver hallazgos).

**Decisión.** Detectar cortes comparando cada frame con el anterior (diferencia
media en grises sobre una miniatura de 64×36 > 18, o correlación de histograma
H-S < 0,5) y recrear el landmarker. Los cortes se escriben en
`<nombre>_cortes.csv`.

**Consecuencia.** El histograma solo no basta: los cortes entre planos de la
misma pista (Ruud) tienen los mismos colores y dan correlación 0,94. La
diferencia en grises los separa: cortes reales ≥ 20, movimiento normal y
barridos ≤ 15. Los fundidos desde negro cuentan como varios cortes seguidos, lo
cual es inocuo. El contrato del CSV de landmarks no cambia.

---

## 2026-10-07 — La corrección va en un CSV aparte

**Contexto.** Las inversiones y los picos se corrigen a posteriori.

**Decisión.** `corregir_esqueleto.py` escribe `<nombre>_corregido.csv` con el
mismo formato más una columna `correccion` (`volteo`, `pico` o vacía). El CSV
original no se toca.

**Consecuencia.** Lo que dijo el modelo y lo que se corrigió quedan separados y
trazables por landmark y frame. Se puede reevaluar cualquier cambio en el
corrector sin volver a pasar el modelo, que es lo caro.

---

## 2026-10-07 — Inversiones: Viterbi de cuerpo entero + ancla de raqueta

**Contexto.** BlazePose voltea el cuerpo entero, no articulaciones sueltas, y la
visibility no lo delata.

**Decisión.** Por frame, dos estados ("tal cual" / "volteado"). Coste de
transición: lo que se mueven hombros, codos, muñecas, caderas, rodillas y
tobillos entre frames, en torsos. Coste de voltear: 0,1 por frame. Ancla: si la
raqueta está a ≤ 0,3 torsos de una muñeca y la otra está al menos 0,8 torsos
más lejos, el estado que deja la mano dominante lejos paga 1,0. La cadena se
corta en frames sin detección, saltos de persona y cortes de escena.

**Consecuencia.** La cara no participa en la decisión (es muy ruidosa de perfil
y de espaldas) pero sí se voltea. La mano dominante es un parámetro (`--lado`);
para zurdos hay que pasarlo. Sin raqueta detectada, un tramo donde el modelo se
equivoque la mayor parte del tiempo queda coherente pero al revés.

---

## 2026-10-07 — La comparación con profesionales es por sesión, no por golpe

**Contexto.** Comparar cada golpe con el p10–p90 de los golpes de referencia
marcaba como "fuera de rango" a un profesional en el 26–44 % de las métricas
(validación dejando fuera a cada jugador).

**Decisión.** Mediana de cada métrica sobre los golpes del clip, comparada con
[mín, máx] de las medianas por jugador de referencia ± 0,25 veces la dispersión
p10–p90 típica de un jugador. Mínimo 3 jugadores por golpe y vista.

**Consecuencia.** Un profesional queda dentro en ~88 % de las métricas. La
herramienta habla de patrones de la sesión ("tus apoyos son más estrechos que
los de cualquier profesional de referencia"), no de golpes sueltos. Con pocos
jugadores de referencia los rangos son anchos y solo se marcan diferencias
claras; es preferible a corregir algo que un profesional también hace.

---

## 2026-10-07 — Métricas del brazo solo con cámara lenta

**Contexto.** A 30 fps el brazo sale movido en el impacto y BlazePose lo
coloca mal sin bajar la visibility. Además, el ángulo del codo cambia 10–12°
con 3 frames de error en el impacto.

**Decisión.** `codo_impacto` e `impacto_delante` se calculan y se guardan, pero
`analizar_clip.py` no los compara por debajo de 100 fps.

**Consecuencia.** Con vídeo normal de móvil, el informe se apoya en piernas,
tronco y apoyos. Para el brazo hay que grabar en cámara lenta.
