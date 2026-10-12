"""Tests de las funciones puras de evaluar_esqueleto, corregir_esqueleto y la
deteccion de cortes de extract_pose.

Usan esqueletos sinteticos: un cuerpo con las articulaciones izquierdas a un
lado y las derechas al otro, que se mueve en linea recta. Sobre esa base se
introducen los errores que se vieron en los clips de referencia (picos,
inversiones de cuerpo entero) y se comprueba que se detectan y se corrigen.
"""

import numpy as np
import pytest

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


def test_angulo_fuera_del_plano_va_de_0_a_180_con_el_signo():
    # hombros de espaldas (izq. a la izq.), de perfil y de frente; ancho real 0.8 torsos
    a = np.zeros((3, 2))
    b = np.array([[0.8, 0], [0.0, 0], [-0.8, 0]])
    assert np.allclose(mg.angulo_fuera_del_plano(a, b, np.ones(3)), [0, 90, 180])


def test_angulo_fuera_del_plano_sin_datos_es_nan():
    a = np.full((3, 2), np.nan)
    assert np.isnan(mg.angulo_fuera_del_plano(a, a, np.ones(3))).all()


def test_hasta_el_extremo_para_cuando_baja_mas_que_la_histeresis():
    v = np.array([0, 1, 2, 1.9, 3, 4, 3.5, 2, 5.0])   # el 1.9 es ruido; el 5 ya es otro gesto
    assert mg.hasta_el_extremo(v, 0, +1, histeresis=1.0) == 5


def test_hasta_el_extremo_hacia_atras_y_saltando_nan():
    v = np.array([0, 3, np.nan, 2, 1, 0.0])
    assert mg.hasta_el_extremo(v, 5, -1, histeresis=1.0) == 1
    assert mg.hasta_el_extremo(np.full(4, np.nan), 0, +1, histeresis=1.0) is None


def serie_de_golpe(n=40, t=20):
    """Golpe sintetico: la muneca barre de -x a +x respecto a la cadera hasta
    el impacto t y luego sube hasta 1.5 hombros; los hombros giran 120 grados."""
    x = np.linspace(-1, 1, n)
    altura = np.r_[np.full(t, 0.5), np.linspace(0.5, 1.5, 10), np.linspace(1.5, 0.5, n - t - 10)]
    giro = np.r_[np.linspace(60, 20, 10), np.linspace(20, 140, n - 10)]
    return {"muneca": np.column_stack([x, np.zeros(n)]), "cadera": np.zeros((n, 2)),
            "pies": np.column_stack([np.full(n, -0.2), np.zeros(n)]), "torso": np.ones(n),
            "altura": altura, "hombros_giro": giro}


def test_direccion_y_peso_delante_siguen_al_golpe():
    s = serie_de_golpe()
    assert mg.direccion_golpe(s, 20) == 1
    assert np.isclose(mg.peso_delante(s, 20), 0.2)       # cadera 0.2 por delante de los pies
    s["muneca"][:, 0] *= -1                                # mismo golpe hacia el otro lado
    assert mg.direccion_golpe(s, 20) == -1
    assert np.isclose(mg.peso_delante(s, 20), -0.2)


def test_terminacion_y_giro_del_golpe_sintetico():
    s = serie_de_golpe()
    assert mg.terminacion(s, 20) == pytest.approx(1.5, abs=0.12)
    # de la maxima torsion (20 grados, frame 9) al final de la serie (140); el
    # suavizado redondea el pico en V y el borde final y le quita unos grados
    assert mg.giro_hombros(s, 20) == pytest.approx(120, abs=10)


def test_terminacion_con_la_muneca_tapada_no_se_mide():
    s = serie_de_golpe()
    s["vis_muneca"] = np.ones(len(s["altura"]))
    assert np.isfinite(mg.terminacion(s, 20))
    s["vis_muneca"][25:] = 0.05            # se tapa mientras aun sube
    assert np.isnan(mg.terminacion(s, 20))
    s["vis_muneca"][:] = 1.0
    s["vis_muneca"][36:] = 0.05            # se tapa cuando ya ha bajado: da igual
    assert mg.terminacion(s, 20) == pytest.approx(1.5, abs=0.12)


def test_hasta_el_extremo_devuelve_none_si_se_oculta_antes():
    v = np.array([0, 1, 2, 3, 4, 3, 1.0])
    oculto = np.zeros(7, bool)
    oculto[3] = True
    assert mg.hasta_el_extremo(v, 0, +1, 1.0, oculto=oculto) is None
    oculto[:] = False
    oculto[6] = True                        # el 3 de despues ya confirma el maximo
    assert mg.hasta_el_extremo(v, 0, +1, 0.5, oculto=oculto) == 4


def test_validacion_dejando_fuera_a_cada_jugador():
    import validar_comparacion as vc
    ref = referencias_sinteticas()
    ref = pd.concat([ref, ref[ref["jugador"] == "j1"].assign(jugador="j4", clip="c4")])
    t = vc.dejar_uno_fuera(ref, fps=30)
    assert set(t["jugador"]) == {"j0", "j1", "j2", "j3", "j4"}
    # los extremos (j0 y j3) quedan fuera del rango de los demas; los del medio, dentro
    assert t.set_index("jugador")["dentro"].to_dict() == {
        "j0": False, "j1": True, "j2": True, "j3": False, "j4": True}


def test_sin_piernas_visibles_no_hay_metricas_de_piernas():
    s = serie_de_golpe()
    n = len(s["torso"])
    s.update(codo=np.full(n, 150.0), rodilla=np.full(n, 160.0), tronco=np.full(n, 5.0),
             apoyos=np.full(n, 0.4), velocidad=np.linspace(0, 1, n), mirada=np.ones(n))
    con = mg.metricas(s, 20, "fondo", 60, vis_brazo=0.9, vis_piernas=0.9)
    sin = mg.metricas(s, 20, "fondo", 60, vis_brazo=0.9, vis_piernas=0.05)
    for clave in mg.METRICAS_PIERNAS:
        assert np.isfinite(con[clave]) and np.isnan(sin[clave])
    assert sin["terminacion"] == con["terminacion"]          # el resto no se toca


def test_el_saque_nunca_cae_en_un_frame_sin_deteccion():
    altura = np.full(100, -1.0)
    altura[20:40] = 1.0
    altura[30] = 3.0                                  # el maximo...
    validos = np.ones(100, bool)
    validos[30] = False                               # ...pero en un frame no valido
    impactos = sg.impactos_de_saque(altura, validos)
    assert len(impactos) == 1 and validos[impactos[0]]


def test_metrica_medida_en_pocos_golpes_lo_dice():
    filas = ac.comparar({"terminacion": np.nan, "rodilla_min": 120.0},
                        referencias_sinteticas(), "derecha", "lateral", 240)
    filas = ac.marcar_pocos_golpes(filas, {"terminacion": 1, "rodilla_min": 7}, 7)
    estados = {f["metrica"]: f["estado"] for f in filas}
    assert estados["terminacion"] == "sin dato (solo se pudo medir en 1 de 7 golpes)"
    assert estados["rodilla_min"] == "dentro del rango"
