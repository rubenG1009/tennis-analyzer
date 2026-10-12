"""Tests de las funciones puras de evaluar_esqueleto, corregir_esqueleto y la
deteccion de cortes de extract_pose.

Usan esqueletos sinteticos: un cuerpo con las articulaciones izquierdas a un
lado y las derechas al otro, que se mueve en linea recta. Sobre esa base se
introducen los errores que se vieron en los clips de referencia (picos,
inversiones de cuerpo entero) y se comprueba que se detectan y se corrigen.
"""

import numpy as np

import corregir_esqueleto as ce
import evaluar_esqueleto as ev
import extract_pose as ep

N = 40
K = len(ce.PARTES_VOLTEO)


def cuerpo_sintetico(n=N):
    """izq/der (n, K, 2): partes izquierdas en x=100, derechas en x=200,
    todo desplazandose 1 px por frame. Torso de 100 px."""
    t = np.arange(n, dtype=float)[:, None, None]
    y = np.linspace(0, 300, K)[None, :, None]
    izq = np.concatenate([np.full((n, K, 1), 100.0) + t, np.broadcast_to(y, (n, K, 1))], axis=2)
    der = izq.copy()
    der[..., 0] += 100
    return izq, der, np.full(n, 100.0)


def voltear(izq, der, frames):
    izq, der = izq.copy(), der.copy()
    izq[frames], der[frames] = der[frames].copy(), izq[frames].copy()
    return izq, der


# --------------------------------------------------------------------------
# Deteccion
# --------------------------------------------------------------------------

def test_movimiento_rectilineo_no_es_pico():
    serie = np.column_stack([np.arange(N) * 30.0, np.zeros(N)])
    assert not ev.detectar_picos(serie, np.full(N, 100.0)).any()


def test_pico_aislado_se_detecta_solo_en_su_frame():
    serie = np.zeros((N, 2))
    serie[10] = [80, 0]
    picos = ev.detectar_picos(serie, np.full(N, 100.0))
    assert picos.tolist() == [i == 10 for i in range(N)]


def test_cruce_izquierda_derecha_se_detecta():
    izq, der, torso = cuerpo_sintetico()
    izq, der = voltear(izq, der, slice(20, None))
    cruces = ev.detectar_cruces(izq[:, 0], der[:, 0], torso)
    assert cruces.tolist() == [i == 20 for i in range(N)]


def test_sin_cruce_en_cuerpo_quieto():
    izq, der, torso = cuerpo_sintetico()
    assert not ev.detectar_cruces(izq[:, 0], der[:, 0], torso).any()


# --------------------------------------------------------------------------
# Correccion de volteos
# --------------------------------------------------------------------------

def test_sin_errores_no_se_voltea_nada():
    izq, der, torso = cuerpo_sintetico()
    assert not ce.decidir_volteos(izq, der, torso, np.zeros(N, bool)).any()


def test_volteo_corto_se_deshace():
    izq, der, torso = cuerpo_sintetico()
    izq, der = voltear(izq, der, slice(15, 18))
    volteo = ce.decidir_volteos(izq, der, torso, np.zeros(N, bool))
    assert volteo.tolist() == [15 <= i < 18 for i in range(N)]


def test_volteo_mayoritario_sin_raqueta_queda_coherente_pero_al_reves():
    # Es la limitacion conocida: sin ancla, gana la orientacion mayoritaria.
    izq, der, torso = cuerpo_sintetico()
    izq, der = voltear(izq, der, slice(5, None))
    volteo = ce.decidir_volteos(izq, der, torso, np.zeros(N, bool))
    assert volteo.tolist() == [i < 5 for i in range(N)]


def test_la_raqueta_ancla_la_orientacion_buena():
    izq, der, torso = cuerpo_sintetico()
    izq, der = voltear(izq, der, slice(5, None))
    # la raqueta esta junto a la muneca derecha real, que en 5.. el modelo llama izquierda
    i_muneca = ce.PARTES_VOLTEO.index("wrist")
    muneca_der_real = cuerpo_sintetico()[1][:, i_muneca]
    cajas = [[(x - 5, y - 5, x + 5, y + 5)] for x, y in muneca_der_real]
    emision = ce.costes_raqueta(cajas, der[:, i_muneca], izq[:, i_muneca], torso)
    volteo = ce.decidir_volteos(izq, der, torso, np.zeros(N, bool), emision_extra=emision)
    assert volteo.tolist() == [i >= 5 for i in range(N)]


def test_la_cadena_no_cruza_un_corte():
    izq, der, torso = cuerpo_sintetico()
    izq, der = voltear(izq, der, slice(20, None))
    cortes = np.zeros(N, bool)
    cortes[20] = True
    # con corte en 20, el tramo 20.. es independiente y se queda tal cual
    assert not ce.decidir_volteos(izq, der, torso, cortes).any()


def test_raqueta_entre_las_dos_manos_no_aporta_evidencia():
    torso = np.full(1, 100.0)
    caja = [[(0, 0, 100, 100)]]
    costes = ce.costes_raqueta(caja, np.array([[50.0, 50]]), np.array([[60.0, 60]]), torso)
    assert costes.tolist() == [[0.0, 0.0]]


def test_raqueta_lejos_de_ambas_manos_no_aporta_evidencia():
    torso = np.full(1, 100.0)
    caja = [[(1000, 1000, 1010, 1010)]]
    costes = ce.costes_raqueta(caja, np.array([[0.0, 0]]), np.array([[200.0, 0]]), torso)
    assert costes.tolist() == [[0.0, 0.0]]


def test_distancia_a_caja():
    caja = (0, 0, 10, 10)
    assert ce.distancia_a_caja((5, 5), caja) == 0
    assert ce.distancia_a_caja((13, 14), caja) == 5


def test_nombre_espejo():
    assert ce.nombre_espejo("left_wrist") == "right_wrist"
    assert ce.nombre_espejo("right_foot_index") == "left_foot_index"
    assert ce.nombre_espejo("nose") == "nose"


# --------------------------------------------------------------------------
# Cortes de escena
# --------------------------------------------------------------------------

def frame_liso(valor, ancho=320, alto=180):
    return np.full((alto, ancho, 3), valor, dtype=np.uint8)


def test_primer_frame_nunca_es_corte():
    assert not ep.es_corte(None, ep.firma_frame(frame_liso(100)))


def test_mismo_frame_no_es_corte():
    f = ep.firma_frame(frame_liso(100))
    assert not ep.es_corte(f, f)


def test_cambio_brusco_de_imagen_es_corte():
    a = ep.firma_frame(frame_liso(30))
    b = ep.firma_frame(frame_liso(200))
    assert ep.es_corte(a, b)


def test_cambio_pequeno_de_iluminacion_no_es_corte():
    a = ep.firma_frame(frame_liso(100))
    b = ep.firma_frame(frame_liso(105))
    assert not ep.es_corte(a, b)


# --------------------------------------------------------------------------
# Segmentacion de golpes
# --------------------------------------------------------------------------

import segmentar_golpes as sg  # noqa: E402
import evaluar_segmentacion as es  # noqa: E402


def test_tramos():
    validos = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert sg.tramos(validos) == [(1, 3), (4, 5), (7, 8)]


def test_un_saque_por_episodio_con_la_muneca_arriba():
    altura = np.zeros(100)
    altura[20:40] = np.r_[np.linspace(0.9, 1.5, 10), np.linspace(1.5, 0.9, 10)]
    assert sg.impactos_de_saque(altura, np.ones(100, bool)) == [29]


def test_episodios_de_saque_cercanos_se_unen():
    altura = np.zeros(100)
    altura[20:30] = 1.2
    altura[33:40] = 1.5          # hueco de 3 frames: es el mismo saque
    assert len(sg.impactos_de_saque(altura, np.ones(100, bool))) == 1


def test_sin_muneca_arriba_no_hay_saque():
    assert sg.impactos_de_saque(np.full(100, 0.5), np.ones(100, bool)) == []


def test_zona_de_lanzamiento_se_dilata_hacia_delante_y_algo_hacia_atras():
    altura_otra = np.zeros(100)
    altura_otra[40:50] = 1.0
    zona = sg.zona_de_lanzamiento(altura_otra, margen=10)
    assert zona[35] and zona[59]
    assert not zona[34] and not zona[60]


def test_pico_de_velocidad_aislado_es_golpe():
    v = np.full(200, 0.01)
    v[100] = 0.5
    assert sg.picos_de_golpe(v, np.ones(200, bool), separacion_min=10) == [100]


def test_ningun_golpe_en_tramo_invalido():
    v = np.full(200, 0.01)
    v[100] = 0.5
    validos = np.ones(200, bool)
    validos[90:110] = False
    assert sg.picos_de_golpe(v, validos, separacion_min=10) == []


def test_emparejar_detecciones():
    pares, libres = es.emparejar([10, 50, 200], [12, 100], ventana=5)
    assert pares == [(12, 10), (100, None)]
    assert libres == [50, 200]


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------

import analizar_clip as ac  # noqa: E402
import metricas_golpe as mg  # noqa: E402
import pandas as pd  # noqa: E402


def test_angulo_recto_y_llano():
    a = np.array([[0.0, 0], [0, 0]])
    b = np.array([[1.0, 0], [1, 0]])
    c = np.array([[2.0, 0], [1, 1]])
    assert np.allclose(mg.angulo(a, b, c), [180, 90])


def test_inclinacion_del_tronco():
    abajo = np.array([[0.0, 10], [0, 10], [0, 10]])
    arriba = np.array([[0.0, 0], [10, 0], [-10, 10]])   # y crece hacia abajo
    assert np.allclose(mg.inclinacion(abajo, arriba), [0, 45, 90])


def test_mediana_ignora_nan():
    serie = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    assert mg.mediana_en(serie, 2, k=2) == 3.0
    assert np.isnan(mg.mediana_en(np.full(5, np.nan), 2))


def referencias_sinteticas():
    filas = [{"clip": f"c{j}", "jugador": f"j{j}", "golpe": "derecha", "vista": "lateral",
              "rodilla_min": 100 + 10 * j + k} for j in range(4) for k in range(5)]
    filas += [{"clip": "s", "jugador": "j0", "golpe": "saque", "vista": "lateral", "rodilla_min": 50}]
    return pd.DataFrame(filas)


def test_rango_por_medianas_de_jugador_con_margen():
    r = ac.rango_referencia(referencias_sinteticas(), "derecha", "lateral", "rodilla_min")
    # medianas por jugador 102, 112, 122, 132; dispersion p10-p90 de cada uno 3.2
    assert r["jugadores"] == 4
    assert np.isclose(r["min"], 102 - ac.MARGEN_INTRA * 3.2)
    assert np.isclose(r["max"], 132 + ac.MARGEN_INTRA * 3.2)


def test_sin_jugadores_suficientes_no_hay_rango():
    assert ac.rango_referencia(referencias_sinteticas(), "saque", "lateral", "rodilla_min") is None
    assert ac.rango_referencia(referencias_sinteticas(), "derecha", "frontal", "rodilla_min") is None


def test_estado_segun_rango():
    ref = referencias_sinteticas()
    estados = [ac.comparar({"rodilla_min": v}, ref, "derecha", "lateral", 30)[0]["estado"]
               for v in (90, 115, 140)]
    assert estados == ["por debajo del rango", "dentro del rango", "por encima del rango"]


def test_metrica_de_brazo_no_se_compara_a_30_fps():
    filas = ac.comparar({"codo_impacto": 150.0}, referencias_sinteticas(), "derecha", "lateral", 30)
    assert "no comparable" in filas[0]["estado"]
