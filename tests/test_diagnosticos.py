"""Tests de diagnosticos: el sentido de cada metrica y la coherencia de la
tabla, que es la parte que se espera que edite alguien que no programa."""

import re

import pytest

import analizar_clip as ac
import diagnosticos as dg


@pytest.mark.parametrize("metrica, estado, esperado", [
    ("giro_hombros", "por debajo del rango", "a revisar"),
    ("giro_hombros", "por encima del rango", "sin problema"),   # girar mas no se corrige
    ("rodilla_min", "por encima del rango", "a revisar"),       # pierna poco flexionada
    ("rodilla_min", "por debajo del rango", "sin problema"),
    ("ancho_apoyos", "por encima del rango", "a revisar"),      # ambos sentidos
    ("ancho_apoyos", "por debajo del rango", "a revisar"),
    ("terminacion", "dentro del rango", "sin problema"),
    ("terminacion", "sin referencias suficientes", None),
    ("codo_impacto", "no comparable a 30 fps (graba en camara lenta)", None),
])
def test_valoracion_segun_sentido(metrica, estado, esperado):
    assert dg.valoracion(metrica, estado) == esperado


def test_diagnostico_solo_si_hay_que_revisar():
    d = dg.diagnostico("giro_hombros", "por debajo del rango", "derecha")
    assert d["problema"].startswith("golpe de brazo")
    assert dg.diagnostico("giro_hombros", "por encima del rango", "derecha") is None
    assert dg.diagnostico("giro_hombros", "dentro del rango", "derecha") is None


def test_diagnostico_depende_del_golpe():
    derecha = dg.diagnostico("terminacion", "por debajo del rango", "derecha")
    reves = dg.diagnostico("terminacion", "por debajo del rango", "reves_2manos")
    assert derecha != reves
    assert dg.diagnostico("terminacion", "por debajo del rango", "saque") is None


def test_toda_metrica_comparada_tiene_sentido():
    assert set(ac.FIABILIDAD) <= set(dg.SENTIDO_PROBLEMA)


def test_la_tabla_es_coherente():
    vistos = set()
    for d in dg.DIAGNOSTICOS:
        assert d["metrica"] in dg.SENTIDO_PROBLEMA
        # no hay diagnostico para el sentido que no es un problema
        assert dg.SENTIDO_PROBLEMA[d["metrica"]] in (dg.AMBOS, d["sentido"])
        assert d["problema"] and d["que_mirar"] and d["causas"] and d["ejercicios"]
        for golpe in d["golpes"]:
            clave = (d["metrica"], d["sentido"], golpe)
            assert clave not in vistos, f"diagnostico duplicado {clave}"
            vistos.add(clave)


def test_toda_metrica_de_derecha_a_revisar_tiene_diagnostico():
    for metrica, sentido in dg.SENTIDO_PROBLEMA.items():
        if metrica == "alcance":
            continue  # solo saque
        for s in ((dg.ABAJO, dg.ARRIBA) if sentido == dg.AMBOS else (sentido,)):
            estado = "por debajo del rango" if s == dg.ABAJO else "por encima del rango"
            assert dg.diagnostico(metrica, estado, "derecha"), (metrica, s)


def test_los_textos_no_llevan_cifras_que_el_verificador_marcaria():
    # redactar_informe avisa de cifras >= 10 que no salen del informe
    for d in dg.DIAGNOSTICOS:
        texto = " ".join([d["problema"], d["que_mirar"], *d["causas"], *d["ejercicios"]])
        assert not [n for n in re.findall(r"\d+", texto) if int(n) >= 10], d["problema"]


def test_comparar_anade_valoracion_y_diagnostico():
    import pandas as pd
    filas = [{"clip": f"c{j}", "jugador": f"j{j}", "golpe": "derecha", "vista": "lateral",
              "giro_hombros": 90 + 5 * j + k} for j in range(4) for k in range(5)]
    ref = pd.DataFrame(filas)
    bajo, alto = (ac.comparar({"giro_hombros": v}, ref, "derecha", "lateral", 30)[0]
                  for v in (40.0, 160.0))
    assert bajo["valoracion"] == "a revisar" and "diagnostico" in bajo
    assert alto["valoracion"] == "sin problema" and "diagnostico" not in alto


def test_en_el_saque_inclinar_el_tronco_no_es_un_defecto():
    assert dg.valoracion("tronco_impacto", "por encima del rango", "saque") == "sin problema"
    assert dg.valoracion("tronco_impacto", "por encima del rango", "derecha") == "a revisar"
    assert dg.diagnostico("tronco_impacto", "por encima del rango", "saque") is None
