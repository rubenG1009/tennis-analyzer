"""Tests de redactar_informe: el prompt, el verificador de cifras y el manejo
de la respuesta. No llaman a la API: el cliente es un doble."""

from types import SimpleNamespace

import pytest

import redactar_informe as ri

INFORME = {
    "video": "data/raw/mi_derecha.mp4", "golpe": "derecha", "vista": "lateral",
    "fps": 30.0, "golpes_detectados": 2, "golpes_descartados_por_tipo": [],
    "sesion": [
        {"metrica": "rodilla_min", "valor": 127.33, "fiabilidad": "alta",
         "descripcion": "flexion maxima de rodillas en la carga",
         "unidad": "grados (180 = pierna recta)", "min": 92.05, "mediana": 120.0,
         "max": 150.03, "jugadores": 7, "golpes": 40, "estado": "dentro del rango"},
        {"metrica": "codo_impacto", "valor": 175.82, "fiabilidad": "baja",
         "descripcion": "extension del codo en el impacto",
         "unidad": "grados (180 = brazo recto)",
         "estado": "no comparable a 30 fps (graba en camara lenta)"},
        {"metrica": "alcance", "valor": None, "fiabilidad": "alta",
         "descripcion": "altura de la muneca sobre la cabeza en el impacto",
         "unidad": "longitudes de torso", "estado": "sin dato (brazo no visible)"},
    ],
    "golpes": [],
}


def test_la_tabla_tiene_una_fila_por_metrica():
    filas = ri.tabla_sesion(INFORME).splitlines()
    assert len(filas) == 2 + len(INFORME["sesion"])


def test_la_tabla_lleva_rango_estado_y_fiabilidad():
    fila = ri.tabla_sesion(INFORME).splitlines()[2]
    assert "127.33" in fila
    assert "92.05 a 150.03 (7 jugadores, 40 golpes)" in fila
    assert "dentro del rango" in fila and "alta" in fila


def test_metrica_sin_valor_ni_rango_sale_con_guiones():
    fila = ri.tabla_sesion(INFORME).splitlines()[4]
    assert "| - | - | sin dato" in fila


def test_el_mensaje_da_el_contexto_del_clip():
    msg = ri.construir_mensaje(INFORME)
    assert "derecha, vista lateral, grabado a 30 fps" in msg
    assert "detectados: 2" in msg


def test_el_mensaje_no_lleva_la_ruta_del_video():
    assert "mi_derecha.mp4" not in ri.construir_mensaje(INFORME)


@pytest.mark.parametrize("texto", [
    "Flexionas hasta 127,3 grados",      # coma decimal y redondeo a 1
    "Flexionas hasta 127.33 grados",     # valor exacto
    "unos 127 grados",                   # redondeo a entero
    "entre 92,1 y 150 grados",           # extremos del rango
    "graba a 120-240 fps",               # cifras del propio prompt
    "en 3 golpes",                       # enteros pequenos se ignoran
])
def test_cifras_respaldadas_no_dan_aviso(texto):
    assert ri.numeros_no_respaldados(texto, INFORME) == []


def test_cifra_inventada_da_aviso():
    texto = "Tu rodilla llega a 127,3 grados, pero tu cadera gira 45,5 grados."
    assert ri.numeros_no_respaldados(texto, INFORME) == ["45,5"]


def _cliente_falso(stop_reason="end_turn", textos=("## Resumen\nBien.",)):
    """Doble de anthropic.Anthropic con lo justo para redactar()."""
    llamadas = []
    respuesta = SimpleNamespace(
        stop_reason=stop_reason, stop_details=None,
        content=[SimpleNamespace(type="thinking", thinking="")]
        + [SimpleNamespace(type="text", text=t) for t in textos])

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return respuesta

    def stream(**kwargs):
        llamadas.append(kwargs)
        return Stream()

    cliente = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return cliente, llamadas


def test_redactar_devuelve_solo_el_texto():
    cliente, _ = _cliente_falso(textos=("## Resumen\n", "Bien."))
    assert ri.redactar(INFORME, cliente) == "## Resumen\nBien."


def test_redactar_pide_el_modelo_con_fallback():
    cliente, llamadas = _cliente_falso()
    ri.redactar(INFORME, cliente)
    (kw,) = llamadas
    assert kw["model"] == ri.MODELO
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"] == ri.SISTEMA
    assert kw["messages"][0]["content"] == ri.construir_mensaje(INFORME)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_redactar_falla_si_la_respuesta_no_esta_completa(stop_reason):
    cliente, _ = _cliente_falso(stop_reason=stop_reason)
    with pytest.raises(RuntimeError):
        ri.redactar(INFORME, cliente)
