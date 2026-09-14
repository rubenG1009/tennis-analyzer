"""Tests dirigidos de scripts/extract_pose.py.

No testean mediapipe ni OpenCV: testean las constantes hardcodeadas, las dos
funciones puras y el contrato estructural del CSV de salida.

El test de integracion usa el modelo LITE (no el heavy) para que tarde
segundos en vez de minutos. Es deliberado: lo que se verifica es la forma del
CSV, y esa no depende de que modelo lo haya rellenado.
"""

import csv
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

import extract_pose as ep


RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_LITE = os.path.join(RAIZ, "models", "pose_landmarker_lite.task")
SCRIPT = os.path.join(RAIZ, "scripts", "extract_pose.py")

COLUMNAS_ESPERADAS = [
    "frame", "t_ms", "landmark_id", "landmark", "x", "y", "z", "visibility",
]


# --------------------------------------------------------------------------
# 1. Integridad de las constantes
# --------------------------------------------------------------------------

def test_hay_exactamente_33_landmarks():
    assert len(ep.NOMBRES_LANDMARKS) == 33


def test_nombres_de_landmark_sin_duplicados():
    assert len(set(ep.NOMBRES_LANDMARKS)) == 33


def test_indices_de_conexiones_en_rango():
    for a, b in ep.CONEXIONES:
        assert 0 <= a <= 32, f"indice fuera de rango: {a}"
        assert 0 <= b <= 32, f"indice fuera de rango: {b}"


def test_no_hay_autoconexiones():
    autos = [(a, b) for a, b in ep.CONEXIONES if a == b]
    assert autos == [], f"conexiones de un landmark consigo mismo: {autos}"


def test_no_hay_conexiones_duplicadas():
    # (a, b) y (b, a) son el mismo hueso: normalizamos antes de comparar.
    normalizadas = [tuple(sorted(par)) for par in ep.CONEXIONES]
    duplicadas = {p for p in normalizadas if normalizadas.count(p) > 1}
    assert duplicadas == set(), f"huesos repetidos: {duplicadas}"


def test_landmarks_del_informe_existen():
    # Si alguien renombra un landmark, el informe se quedaria mudo en silencio.
    for nombre in ep.LANDMARKS_INFORME:
        assert nombre in ep.NOMBRES_LANDMARKS


# --------------------------------------------------------------------------
# 2. Timestamps
# --------------------------------------------------------------------------

def serie_t_ms(n_frames, fps):
    """Reproduce el bucle del script: encadena calcular_t_ms n veces."""
    serie, ultimo = [], -1
    for idx in range(n_frames):
        ultimo = ep.calcular_t_ms(idx, fps, ultimo)
        serie.append(ultimo)
    return serie


@pytest.mark.parametrize("fps", [24, 25, 30, 50, 60, 120, 240])
def test_t_ms_estrictamente_creciente(fps):
    serie = serie_t_ms(500, fps)
    diffs = np.diff(serie)
    assert (diffs > 0).all(), f"t_ms no crece con fps={fps}"


@pytest.mark.parametrize("fps", [24, 25, 30, 50, 60, 120, 240])
def test_t_ms_respeta_la_formula_con_fps_normales(fps):
    # Con fps <= 1000 la guarda no debe entrar nunca: el valor tiene que ser
    # exactamente round(idx * 1000 / fps).
    serie = serie_t_ms(500, fps)
    esperado = [round(i * 1000 / fps) for i in range(500)]
    assert serie == esperado


def test_a_1000_fps_la_formula_todavia_se_basta():
    # Caso limite exacto: round(idx * 1000 / 1000) == idx, que ya crece de 1
    # en 1. La guarda no llega a activarse.
    serie = serie_t_ms(500, 1000)
    assert serie == list(range(500))


def test_la_guarda_actua_con_fps_patologico():
    fps = 2000  # por encima de 1000 el redondeo empieza a repetir valores
    crudo = [round(i * 1000 / fps) for i in range(500)]
    serie = serie_t_ms(500, fps)

    # (a) sin guarda la serie SI se repetiria: el caso patologico es real
    assert len(set(crudo)) < len(crudo)
    # (b) con guarda, monotonia estricta (que es lo que exige mediapipe)
    assert (np.diff(serie) > 0).all()
    # (c) y la guarda ha intervenido de verdad, no es un no-op
    assert serie != crudo


# --------------------------------------------------------------------------
# 3. Contrato del CSV (integracion con video sintetico)
# --------------------------------------------------------------------------

N_FRAMES_SINTETICO = 20


@pytest.fixture
def video_sintetico(tmp_path):
    """Video plano de 20 frames sin ninguna persona."""
    ruta = str(tmp_path / "sintetico.mp4")
    escritor = cv2.VideoWriter(
        ruta, ep.obtener_fourcc("mp4v"), 30.0, (256, 256))
    assert escritor.isOpened(), "OpenCV no puede escribir el video de prueba"
    for i in range(N_FRAMES_SINTETICO):
        frame = np.full((256, 256, 3), 40 + i * 3, dtype=np.uint8)
        cv2.rectangle(frame, (60, 60), (190, 190), (120, 120, 120), -1)
        escritor.write(frame)
    escritor.release()
    return ruta


@pytest.mark.skipif(not os.path.isfile(MODELO_LITE),
                    reason=f"falta el modelo lite en {MODELO_LITE}")
def test_contrato_estructural_del_csv(video_sintetico, tmp_path):
    salida = ep.procesar_video(video_sintetico, MODELO_LITE, str(tmp_path))

    with open(salida["ruta_csv"], newline="") as f:
        filas = list(csv.reader(f))

    cabecera, datos = filas[0], filas[1:]

    # columnas exactas y en orden
    assert cabecera == COLUMNAS_ESPERADAS

    # una fila por landmark y frame, sin excepcion
    assert len(datos) == N_FRAMES_SINTETICO * 33

    frames = [int(fila[0]) for fila in datos]
    assert sorted(set(frames)) == list(range(N_FRAMES_SINTETICO)), \
        "hay huecos en el eje temporal"

    # los 33 landmarks en cada frame, con sus ids y nombres correctos
    for n in range(N_FRAMES_SINTETICO):
        del_frame = [fila for fila in datos if int(fila[0]) == n]
        assert len(del_frame) == 33
        assert [int(fila[2]) for fila in del_frame] == list(range(33))
        assert [fila[3] for fila in del_frame] == ep.NOMBRES_LANDMARKS

    # t_ms unico por frame y estrictamente creciente
    t_por_frame = {}
    for fila in datos:
        t_por_frame.setdefault(int(fila[0]), set()).add(int(fila[1]))
    assert all(len(v) == 1 for v in t_por_frame.values()), \
        "un mismo frame tiene varios t_ms"
    serie = [t_por_frame[n].pop() for n in range(N_FRAMES_SINTETICO)]
    assert (np.diff(serie) > 0).all()


@pytest.mark.skipif(not os.path.isfile(MODELO_LITE),
                    reason=f"falta el modelo lite en {MODELO_LITE}")
def test_las_celdas_estan_llenas_o_vacias_pero_nunca_a_medias(
        video_sintetico, tmp_path):
    # No asumimos si mediapipe detecta algo o no en el video sintetico: lo que
    # no puede pasar es una fila con x pero sin visibility, o al reves.
    salida = ep.procesar_video(video_sintetico, MODELO_LITE, str(tmp_path))
    with open(salida["ruta_csv"], newline="") as f:
        datos = list(csv.reader(f))[1:]

    for fila in datos:
        valores = fila[4:8]  # x, y, z, visibility
        vacios = [v == "" for v in valores]
        assert all(vacios) or not any(vacios), f"fila a medio rellenar: {fila}"


@pytest.mark.skipif(not os.path.isfile(MODELO_LITE),
                    reason=f"falta el modelo lite en {MODELO_LITE}")
def test_genera_el_video_de_debug(video_sintetico, tmp_path):
    salida = ep.procesar_video(video_sintetico, MODELO_LITE, str(tmp_path))
    assert os.path.isfile(salida["ruta_debug"])
    captura = cv2.VideoCapture(salida["ruta_debug"])
    assert captura.isOpened()
    captura.release()


# --------------------------------------------------------------------------
# 4. Informe de calidad (funcion pura)
# --------------------------------------------------------------------------

def test_estadisticas_sin_valores():
    media, minima, pct = ep.calcular_estadisticas_visibility([], 10)
    assert media is None
    assert minima is None
    assert pct == 0.0


def test_estadisticas_todos_por_encima():
    media, minima, pct = ep.calcular_estadisticas_visibility(
        [0.8, 0.9, 1.0], 3)
    assert media == pytest.approx(0.9)
    assert minima == pytest.approx(0.8)
    assert pct == pytest.approx(100.0)


def test_estadisticas_todos_por_debajo():
    media, minima, pct = ep.calcular_estadisticas_visibility([0.1, 0.2], 2)
    assert media == pytest.approx(0.15)
    assert minima == pytest.approx(0.1)
    assert pct == pytest.approx(0.0)


def test_estadisticas_caso_mixto_calculado_a_mano():
    # valores: 0.9, 0.4, 0.6, 0.1 sobre un video de 5 frames leidos.
    # media   = (0.9 + 0.4 + 0.6 + 0.1) / 4 = 0.5
    # minima  = 0.1
    # >= 0.5  = {0.9, 0.6} = 2 de 5 frames leidos = 40 %
    media, minima, pct = ep.calcular_estadisticas_visibility(
        [0.9, 0.4, 0.6, 0.1], 5)
    assert media == pytest.approx(0.5)
    assert minima == pytest.approx(0.1)
    assert pct == pytest.approx(40.0)


def test_el_umbral_incluye_el_valor_exacto():
    # Criterio unificado a >=: un 0.5 clavado SI cuenta, igual que
    # dibujar_esqueleto lo pinta verde. Antes el informe usaba > y el dibujado
    # >=, lo que hacia que un mismo landmark saliera verde en el video pero no
    # contara en el porcentaje.
    _, _, pct = ep.calcular_estadisticas_visibility([0.5, 0.5], 2)
    assert pct == pytest.approx(100.0)


def test_informe_y_dibujado_comparten_el_mismo_umbral():
    # Ambos leen UMBRAL_VISIBILITY. Si alguien vuelve a separar los criterios,
    # o cambia el operador en uno solo de los dos sitios, esto falla.
    justo = ep.UMBRAL_VISIBILITY
    apenas_debajo = ep.UMBRAL_VISIBILITY - 1e-9
    _, _, pct_justo = ep.calcular_estadisticas_visibility([justo], 1)
    _, _, pct_debajo = ep.calcular_estadisticas_visibility([apenas_debajo], 1)
    assert pct_justo == pytest.approx(100.0), "el valor exacto del umbral debe contar"
    assert pct_debajo == pytest.approx(0.0), "por debajo del umbral no debe contar"


def test_el_porcentaje_se_mide_sobre_los_frames_leidos_no_sobre_los_detectados():
    # 2 valores, ambos por encima, pero el video tenia 10 frames: 20 %, no 100.
    _, _, pct = ep.calcular_estadisticas_visibility([0.9, 0.8], 10)
    assert pct == pytest.approx(20.0)


def test_estadisticas_no_dividen_por_cero_sin_frames():
    media, minima, pct = ep.calcular_estadisticas_visibility([0.9], 0)
    assert media == pytest.approx(0.9)
    assert pct == 0.0


# --------------------------------------------------------------------------
# 5. Manejo de errores del CLI
# --------------------------------------------------------------------------

def ejecutar_cli(*args):
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True)


def test_video_inexistente_sale_con_error_claro(tmp_path):
    proceso = ejecutar_cli(str(tmp_path / "no_existe.mp4"),
                           "--model", MODELO_LITE,
                           "--out-dir", str(tmp_path))
    salida = proceso.stdout + proceso.stderr
    assert proceso.returncode != 0
    assert "ERROR: no existe el video" in salida
    assert "Traceback" not in salida


def test_modelo_inexistente_sale_con_error_claro(video_sintetico, tmp_path):
    proceso = ejecutar_cli(video_sintetico,
                           "--model", str(tmp_path / "no_existe.task"),
                           "--out-dir", str(tmp_path))
    salida = proceso.stdout + proceso.stderr
    assert proceso.returncode != 0
    assert "ERROR: no existe el modelo" in salida
    assert "Traceback" not in salida
