import io
import json
import urllib.error

import cv2
import numpy as np
import pytest

import combate_d1 as d1


def respuesta():
    return {"answers": {k: {"noul": p} for k, p in zip(d1.PREGUNTAS, (0.8, 0.2, 0.6))}}


def test_consulta_envia_imagen_y_preguntas_con_esquema_liquid():
    def abrir(req, timeout):
        datos = json.loads(req.data)
        assert req.full_url == d1.URL and timeout == 45
        assert req.get_header("Authorization") == "Bearer clave-prueba"
        assert datos["model"] == "d1-ultra"
        assert datos["images"] == ["data:image/jpeg;base64,aW1hZ2Vu"]
        assert set(datos["questions"]) == set(d1.PREGUNTAS)
        return io.BytesIO(json.dumps(respuesta()).encode())
    assert d1.consultar(b"imagen", "clave-prueba", abrir)["cuerpo_entero"] == 0.8


@pytest.mark.parametrize("datos", [{}, {"answers": {k: {"noul": 2} for k in d1.PREGUNTAS}}])
def test_rechaza_respuesta_incompleta_o_probabilidad_fuera_de_rango(datos):
    with pytest.raises(d1.ErrorD1, match="incompletas o inválidas"):
        d1.consultar(b"imagen", "clave", lambda *a, **kw: io.BytesIO(json.dumps(datos).encode()))


def test_error_http_no_revela_clave():
    def abrir(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "clave-secreta", {}, None)
    with pytest.raises(d1.ErrorD1) as error:
        d1.consultar(b"imagen", "clave-secreta", abrir)
    assert "401" in str(error.value)
    assert "clave-secreta" not in str(error.value)


def test_revisar_video_real_muestrea_y_guarda_sin_imagenes(tmp_path, monkeypatch):
    video = tmp_path / "prueba.avi"
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert escritor.isOpened()
    for i in range(20):
        escritor.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    escritor.release()
    imagenes = []
    def simular(imagen, clave):
        assert clave == "clave-prueba"
        imagenes.append(imagen)
        return {k: 0.5 for k in d1.PREGUNTAS}
    monkeypatch.setattr(d1, "consultar", simular)
    resultado = d1.revisar(video, 4, "clave-prueba")
    assert [f["frame"] for f in resultado["fotogramas"]] == [2, 7, 12, 17]
    assert len(imagenes) == 4 and all(i.startswith(b"\xff\xd8") for i in imagenes)
    ruta = tmp_path / "resultado.json"
    d1.guardar(resultado, ruta)
    guardado = json.loads(ruta.read_text())
    assert guardado["video_sha256"] == d1.huella(video)
    assert "images" not in ruta.read_text()


def test_sin_clave_no_extrae_ni_envia_fotogramas(monkeypatch):
    monkeypatch.delenv("LIQUID_API_KEY", raising=False)
    with pytest.raises(d1.ErrorD1, match="LIQUID_API_KEY"):
        d1.revisar("no-existe.mp4")
