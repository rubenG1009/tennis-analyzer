# Hallazgos

Observaciones sobre clips reales. Una entrada por clip o por sesión de análisis.

---

## 2026-09-14 — `prueba.mp4`: el pipeline funciona, el clip no sirve para la muñeca

### El clip

| | |
|---|---|
| Fichero | `data/raw/prueba.mp4` |
| Resolución | 360 × 640 (vertical) |
| FPS | 30 |
| Frames | 355 declarados, 355 leídos |
| Duración | 11,83 s |
| Origen | Descargado de YouTube |
| Encuadre | Jugador lejos de cámara y de espaldas, pista completa en plano |

### Resultado

Modelo heavy, 355 frames, **0 frames sin detección (0,00 %)**. MediaPipe
encuentra una pose en todos y cada uno de los frames.

Pero encontrar una pose no es lo mismo que encontrarla bien:

| Landmark | visibility media | mínima | % ≥ 0.5 |
|---|---|---|---|
| `right_shoulder` | 1.000 | 1.000 | 100,00 % |
| `right_hip` | 1.000 | 1.000 | 100,00 % |
| `right_knee` | 0.860 | 0.203 | 94,08 % |
| `right_elbow` | 0.605 | 0.162 | 67,04 % |
| `right_wrist` | 0.383 | 0.007 | 35,77 % |

El gradiente es muy claro y va de dentro a fuera: hombro y cadera perfectos,
rodilla bien, codo regular, muñeca mal. Cuanto más distal la articulación, peor.
Se confirma a ojo en el vídeo de debug: en el frame 120 los puntos rojos caen
justo sobre el brazo que sostiene la raqueta.

### Conclusión

**El pipeline funciona. El dato de origen no sirve para métricas de muñeca.**

Es una limitación de la fuente, no del código. El jugador está de espaldas, lejos
y a 360 × 640: el brazo que golpea queda ocluido por el propio cuerpo durante
buena parte del swing, y la muñeca ocupa unos pocos píxeles. Ningún
posprocesamiento arregla eso. Suavizar una señal que está ausente el 64 % del
tiempo no la recupera, solo fabrica una curva plausible sobre huecos, que es
peor que no tener nada porque parece un dato.

No tiene sentido construir métricas de golpeo sobre este clip.

### Implicación práctica

Criterio de aceptación para los clips que se graben a propósito:

> **`right_wrist` debe alcanzar el umbral de `visibility` en más del 90 % de los
> frames del golpe.**

Se mide sobre los frames del golpe, no sobre el clip entero: lo que importa es la
ventana en la que se calcularán las métricas, no los segundos de jugador parado.

De lo observado aquí se deduce cómo grabar: jugador **de perfil** respecto a la
cámara (no de espaldas), cerca, y con el lado que golpea hacia el objetivo. La
resolución importa menos que el encuadre — subir a 1080p no ayuda si la muñeca
sigue tapada por el torso.

### Verificación de que los tests muerden

Antes de fiarse de la suite se comprobó que falla cuando debe, rompiendo el
script a propósito y restaurándolo después (verificando por hash que quedaba
intacto):

| Mutación introducida | Resultado |
|---|---|
| Quitar la guarda de monotonía de `t_ms` | `test_la_guarda_actua_con_fps_patologico` FAILED |
| Cambiar `>=` por `>` en el umbral | `test_el_umbral_es_estricto...` FAILED |
| Quitar un landmark (32 en vez de 33) | 3 tests FAILED |
| Meter una autoconexión `(7,7)` en `CONEXIONES` | `test_no_hay_autoconexiones` FAILED |
| Omitir las filas de los frames sin detección | `test_contrato_estructural_del_csv` FAILED |

La última es la que más importa: es el contrato de "el eje temporal no puede
tener huecos silenciosos", y falla ruidosamente si alguien lo rompe.

Nota: la mutación del umbral se ejecutó cuando el criterio del informe era `>` y
el test fijaba esa asimetría. Tras unificar ambos a `>=`, el test equivalente es
`test_el_umbral_incluye_el_valor_exacto`, más `test_informe_y_dibujado_comparten_el_mismo_umbral`,
que falla si alguien vuelve a separar los dos criterios.

---

## 2026-10-07 — Clips de referencia: dónde falla el esqueleto y cuánto se arregla

### Los clips

12 clips de profesionales bajados de YouTube con `scripts/download_references.py`
(lista en `references/manifest.csv`): 6 derechas, 3 reveses y 3 saques; vista
lateral, frontal y de espaldas; de 320×240 a 1920×1080, de 24 a 60 fps;
17 545 frames en total. Todos diestros.

### Cómo se mide sin anotaciones

No hay ground truth, así que `scripts/evaluar_esqueleto.py` busca síntomas
medibles en la propia serie: **picos** (un landmark se va lejos un frame y
vuelve), **cruces** (izquierda y derecha intercambiadas entre frames
consecutivos) y **saltos de persona** (la cadera se teletransporta o el torso
cambia de escala). Todo en longitudes de torso. Cada tipo se validó mirando los
frames marcados antes de fiarse de él.

### Los tres errores reales

**1. Inversión izquierda/derecha del cuerpo entero.** El más grave y el más
traicionero. BlazePose duda entre "mira a cámara" y "está de espaldas" y, cuando
duda, voltea el cuerpo entero: hombros, caderas, brazos y piernas cambian de
lado a la vez. La visibility **no baja** (Serena de espaldas: muñeca ≥ 0.5 en el
98 % de los frames, nariz a 0.99 aunque no se le ve la cara). Peor aún: a veces
la inversión se sostiene decenas de frames (saque de Roddick, frames 473 en
adelante: el brazo que lanza la bola etiquetado como derecho).

> **La visibility alta no garantiza que el landmark sea el que dice ser.**

**2. Persona equivocada tras un corte de escena.** En modo VIDEO MediaPipe sigue
la región del frame anterior. Tras un corte (revés de Dimitrov, frame 2322) se
quedó 35 frames sobre un jugador del fondo en vez de sobre Dimitrov, que estaba
en primer plano.

**3. Picos de un frame.** Fallos puntuales, sobre todo en la muñeca. Además, un
fundido desde negro (revés de Djokovic, frames 3–15) produce esqueletos absurdos.

### Qué se hizo

| Arreglo | Dónde |
|---|---|
| Detectar cortes de escena y recrear el `PoseLandmarker` | `extract_pose.py` |
| Detectar la raqueta en cada frame (EfficientDet-Lite2, COCO) | `extract_pose.py --raqueta` |
| Deshacer inversiones con Viterbi de 2 estados + ancla de raqueta | `corregir_esqueleto.py` |
| Sustituir picos por el punto medio de sus vecinos | `corregir_esqueleto.py` |

### Resultado

% de frames con algún síntoma, mismo evaluador antes y después:

| Clip | Antes | Después |
|---|---|---|
| `sv_serena` (espaldas) | 16,47 % | 8,23 % |
| `sv_roddick` | 5,18 % | 1,52 % |
| `sv_zverev_lat` | 2,00 % | 0,80 % |
| `bh_dimitrov_lat` | 1,71 % | 0,69 % |
| `bh_djokovic_1000fps` | 1,63 % | 0,59 % |
| `fh_roddick_lat` | 1,50 % | 0,37 % |
| `fh_ruud` | 0,93 % | 0,97 % |
| resto (5 clips) | ≤ 0,46 % | ≤ 0,21 % |
| **Media** | **2,52 %** | **1,12 %** |

Cruces: 1,51 % → 0,77 %. Picos: 1,59 % → 0,40 %. Los cinco clips que ya estaban
limpios no reciben ningún volteo.

Se verificó a ojo en los casos que motivaron cada arreglo: Roddick 472–475 y
Serena 252–255 y 383 quedan con `R` en la mano de la raqueta.

### Lo que salió mal por el camino (y por qué importa)

- **Sin ancla, la continuidad acierta "coherentemente al revés".** La primera
  versión del corrector solo buscaba continuidad. En Serena funcionó; en Roddick
  dejó los 4 frames coherentes pero con la mano de la raqueta etiquetada como
  izquierda, porque en ese tramo el modelo se equivocaba la mayoría del tiempo.
  Hay un test que fija esta limitación.
- **Un ancla de raqueta laxa rompe clips buenos.** Con distancia a la caja
  ≤ 1 torso y margen 0,3, la derecha de Dimitrov recibió 236 volteos de 599: en
  la posición de espera las dos manos están en la raqueta, y en la terminación
  la raqueta acaba sobre el hombro contrario y la recoge la otra mano. Se
  endureció a "una muñeca a ≤ 0,3 torsos y la otra 0,8 más lejos" (agarre a una
  mano) y los volteos en clips limpios bajaron a cero.
- **Un pico marcaba también a sus vecinos**, que se desvían la mitad. Corregirlos
  habría estropeado dos frames buenos por cada malo. Ahora solo cuenta el máximo
  local; lo encontró un test.

### Lo que queda abierto

- **La muñeca derecha en el revés está mal vista incluso en buenas condiciones**:
  51–61 % de frames ≥ 0.5 en los tres reveses, frente a 72–88 % en derechas
  laterales. En el revés a dos manos la muñeca derecha queda detrás de la
  izquierda. El criterio de aceptación del 90 % (entrada del 2026-09-14) no lo
  cumple ningún revés: para métricas de revés habrá que usar la muñeca más
  visible, o las dos.
- Serena sigue en un 8 %: de espaldas la ambigüedad es máxima y la raqueta a
  menudo queda tapada por el cuerpo.
- El evaluador mide síntomas, no exactitud. Un esqueleto que está mal de forma
  suave y continua no lo detecta. Para medir exactitud haría falta anotar a mano
  algunos frames.

---

## 2026-10-07 (2) — Segmentación, métricas y comparación con profesionales

### Datos

- **21 clips, 14 jugadores identificados** (más una compilación de varios). A los 12 de la primera tanda se añadieron 9 clips
  laterales a velocidad real (30–60 fps): Sinner, Alcaraz ×2, Fils, Djokovic,
  Lehecka, Zheng, Ymer, Nava.
- **`references/golpes.csv`: 51 impactos anotados a mano.** Los 22 de Sinner
  están verificados frame a frame (±2 frames).
- **`references/golpes_auto.csv`: 82 golpes** cuyo impacto puso el
  segmentador y cuyo tipo (derecha/revés) verifiqué a ojo. No se usan para
  evaluar al segmentador, que sería circular.
- **Excluidos:** Zheng (cámara elevada por detrás, no comparable) y los 5 reveses
  cortados de Alcaraz. Fils está descargado y procesado pero sin etiquetar.

### Segmentación (`segmentar_golpes.py`)

Sobre los impactos anotados a mano:

| Material | Encontrados | Precisión | Impacto dentro de tolerancia |
|---|---|---|---|
| Velocidad real (33 golpes) | 91 % | 86 % | 26/30 |
| Cámara lenta editada (18 golpes) | 78 % | 54 % | 10/14 |
| Saques (5) | 5/5 | — | 5/5 |

En Sinner el error del impacto es de **0 frames de mediana** (19 de 20 a ±2).
La cámara lenta editada (con rampas de velocidad) es el caso malo: el pico de
velocidad de la muñeca cae en la terminación, no en el impacto. Para vídeo de
móvil, que es lo que se va a usar, no importa.

Qué hizo falta para llegar ahí:
- **Un bug de valor por defecto** (`prominencia_rel=PROMINENCIA_RELATIVA` se
  evaluaba al definir la función) hizo que el primer barrido de parámetros
  diera el mismo resultado para todos los valores. Lo delató precisamente eso.
- **Prominencia 6 → 3.** Con 6 solo se encontraba el 27 % a velocidad real.
- **El filtro de raqueta se desactiva si el detector apenas la ve** (< 50 % de
  frames). En Sinner la ve en el 42 % y el filtro eliminaba los 22 golpes.
- **El saque se detecta por geometría, no por velocidad:** muñeca dominante en
  su punto más alto sobre la cabeza. Los picos durante el lanzamiento de bola
  se descartan.
- **Derecha frente a revés no se clasifica.** La distancia entre muñecas en el
  impacto no separa el revés a dos manos (derechas con 0,02–0,42 torsos, el
  mismo rango). **Un clip, un tipo de golpe.**

### Métricas (`metricas_golpe.py`)

Sensibilidad a un error de ±3 frames en el impacto (cambio mediano):

| Métrica | Cambio | Fiabilidad asignada |
|---|---|---|
| rodilla_min | 0 | alta |
| alcance (saque) | 0,02 torsos | alta |
| ancho_apoyos | 0,08 torsos | alta |
| tronco_impacto | 2° | alta |
| altura_impacto | 0,17 | media |
| impacto_delante | 0,35–0,6 torsos | baja |
| codo_impacto | **10–12°** | baja |

**El brazo sale mal colocado en el impacto a 30 fps.** En el impacto el brazo
va a máxima velocidad, sale movido en la imagen y BlazePose dibuja el codo
doblado pegado al cuerpo, con visibility ~0,6 (no se marca como dudoso).
Verificado a ojo en Sinner 272 y 448. Medido con la raqueta detectada: la
muñeca queda de mediana a 0,48 torsos de la raqueta en el impacto a 30 fps,
frente a 0,11 en cámara lenta. En Djokovic a 1000 fps el brazo está bien.
Por eso `analizar_clip.py` no compara codo ni impacto por delante por debajo de
100 fps.

> **Para grabar: cámara lenta del móvil (120 o 240 fps).** Es lo que más
> mejora los datos, más que la resolución.

### Comparación con los profesionales (`analizar_clip.py`)

Validación: analizar a cada profesional como si fuera un usuario, con él fuera
de las referencias. Un sistema bien calibrado debería dejarlo "dentro del
rango" casi siempre.

| Cómo se construye el rango | Pro dentro del rango |
|---|---|
| p10–p90 de todos los golpes, primera tanda (pocos jugadores) | 56 % |
| p10–p90 de todos los golpes, con 6 jugadores en derecha lateral | 74 % |
| Por sesión: medianas por jugador ± 0,25 × dispersión de un jugador | **~88 %** |

La conclusión de fondo: **las diferencias entre profesionales son mayores que
entre golpes de un mismo profesional.** Con 2–3 jugadores, el "rango
profesional" es el estilo de esos jugadores, y marcaría como error cosas que
otros profesionales hacen. Por eso se exige un mínimo de 3 jugadores por golpe
y vista, y hoy solo se cumple en:

| Golpe / vista | Jugadores |
|---|---|
| derecha / lateral | 6 |
| derecha / frontal | 4 |
| revés 2 manos / lateral | 4 |
| saque (cualquier vista) | 1 — **no se compara** |
| derecha / espalda | 1 — **no se compara** |

Los rangos que salen son amplios (rodilla 92–150° en derecha lateral): el
sistema solo marca diferencias claras. Ejemplo real: Ymer, con él fuera de las
referencias, sale con los apoyos más estrechos que todos los demás (0,76
frente a ≥ 0,97), y es una diferencia que se ve en el vídeo.

### Lo que queda abierto

- Saque y vista de espaldas: faltan jugadores de referencia.
- Clasificar derecha/revés automáticamente.
- La vista "lateral" de estos clips significa cámara en el lateral de la pista:
  el jugador queda de frente en la derecha y de espaldas en el revés (o al
  revés, según el lado). Hay que grabar igual para que la comparación valga.
