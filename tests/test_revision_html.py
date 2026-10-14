"""Tests del generador de informes de revision: las funciones que preparan
los datos. La plantilla HTML se comprueba renderizandola, no aqui."""

import numpy as np
import pandas as pd

import revision_html as rh


def test_tramos_de_true_con_fin_incluido():
    assert rh.tramos([0, 1, 1, 0, 1, 0, 0, 1]) == [[1, 2], [4, 4], [7, 7]]
    assert rh.tramos([]) == []


def test_reducir_toma_el_minimo_de_cada_bloque_y_respeta_nan():
    v = np.r_[np.ones(10), 0.2, np.ones(9), np.full(10, np.nan)]
    r = rh.reducir(v, n=3)
    assert r == [1.0, 0.2, None]          # la caida a 0.2 no se pierde al reducir
    assert rh.reducir(np.ones(5), n=100) == [1.0] * 5


def test_eventos_de_correccion_agrupa_volteos_en_tramos():
    corr = pd.DataFrame({"frame": [3, 3, 4, 5, 9, 12], "landmark_id": [11, 12, 11, 11, 11, 16],
                         "correccion": ["volteo", "volteo", "volteo", "volteo", "volteo", "pico"]})
    volteos, picos = rh.eventos_de_correccion(corr)
    assert volteos == [[3, 5], [9, 9]]
    assert picos == [12]


def test_huesos_solo_entre_puntos_que_se_exportan():
    assert all(0 <= a < len(rh.PUNTOS) and 0 <= b < len(rh.PUNTOS) for a, b in rh.HUESOS)
    # hombro-codo derecho (12-14) esta; ojo-ojo (1-2) no
    assert (rh.PUNTOS.index(12), rh.PUNTOS.index(14)) in rh.HUESOS
    assert 1 not in rh.PUNTOS


def clip_base(**cambios):
    clip = {"deteccion_pct": 97.0, "visible_pct": {"brazo dominante": 80.0, "piernas": 90.0},
            "fps": 240.0, "golpes": [{}] * 7, "con_golpes": True, "sesion": [], "impactos": "automaticos"}
    clip.update(cambios)
    return clip


def test_sin_problemas_no_hay_avisos():
    assert rh.avisos(clip_base()) == []


def test_avisos_de_lo_que_hay_que_saber():
    a = rh.avisos(clip_base(deteccion_pct=60.0, visible_pct={"brazo dominante": 30.0, "piernas": 5.0},
                            fps=30.0, golpes=[{}] * 2, impactos="a mano",
                            sesion=[{"metrica": "terminacion", "estado": "sin dato (x)"}]))
    texto = " ".join(a)
    assert "60 %" in texto and "Brazo dominante" in texto and "Piernas" in texto
    assert "30 fps" in texto and "a mano" in texto and "Solo 2 golpes" in texto
    assert "terminacion: sin dato" in texto


def test_limpiar_deja_json_valido():
    import json
    v = rh.limpiar({"a": np.nan, "b": [np.float64(1.5), np.inf], "c": np.int64(3)})
    assert v == {"a": None, "b": [1.5, None], "c": 3}
    json.dumps(v, allow_nan=False)


def test_resumen_para_el_indice():
    clip = clip_base(nombre="x", golpe="derecha", vista="lateral", duracion=31.0,
                     volteos=[[1, 3], [10, 10]], picos=[5, 6], avisos=["a"],
                     sesion=[{"metrica": "giro_hombros", "valoracion": "a revisar"},
                             {"metrica": "rodilla_min", "valoracion": "sin problema"}])
    r = rh.resumen(clip)
    assert r["volteos"] == 4 and r["picos"] == 2 and r["a_revisar"] == ["giro_hombros"]
    assert r["grupo"] == "propio" and r["titulo"] == "x" and r["comparadas"] == 2


def test_golpe_representativo_es_el_mas_cercano_a_la_mediana():
    golpes = [{"giro_hombros": 60.0}, {"giro_hombros": 90.0}, {"giro_hombros": 84.0},
              {"giro_hombros": None}, {"giro_hombros": 70.0}, {"giro_hombros": 76.0}]
    assert rh.representativo(golpes) == 5          # mediana de 60, 70, 76, 84, 90 = 76
    assert rh.representativo([{"giro_hombros": None}]) == 0
    assert rh.representativo([]) is None


def test_consejos_de_grabacion_salen_de_lo_que_fallo():
    assert rh.consejos_grabacion(clip_base(vista="lateral", golpes=[{}] * 12)) == []
    c = rh.consejos_grabacion(clip_base(
        vista="frontal", grupo="propio", fps=30.0, impactos="a mano", golpes=[{}] * 7,
        visible_pct={"brazo dominante": 80.0, "piernas": 5.0},
        sesion=[{"metrica": "terminacion", "estado": "sin dato (solo se pudo medir en 1 de 7 golpes)"}]))
    texto = " ".join(c)
    for clave in ("piernas", "45°", "cámara lenta", "bola", "10 a 15", "de lado"):
        assert clave in texto, clave
