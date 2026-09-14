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
