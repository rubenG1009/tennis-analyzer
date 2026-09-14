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
