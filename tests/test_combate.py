"""Tests del analisis de combate (boxeo / K-1) con un luchador sintetico.

El luchador mira hacia +x, de torso 100 px (hombros y=100, cadera y=200), guardia
con las manos junto a la cara (y=85), tobillos en y=370. Sobre eso se montan
jabs, patadas, pasos que NO son patadas y una guardia que baja, y se comprueba
que se detectan, se miden en unidades reales y se priorizan segun el perfil.
"""

import numpy as np
import pytest

import analizar_combate as ac
import combate_correcciones as cc
import combate_golpes as cg
import combate_metricas as cm
import combate_perfil as cp
import combate_referencias as cr

FPS = 120
N = 720
PERFIL = cp.Perfil(altura_cm=180, peso_kg=75, experiencia_meses=12)


def bump(n, inicio, ancho, alto):
    """Pulso suave de `alto` que empieza en `inicio` y dura `ancho` frames."""
    t = np.arange(n)
    x = (t - inicio) / ancho
    return np.where((x >= 0) & (x <= 1), alto * np.sin(np.pi * x) ** 2, 0.0)


def luchador(jabs=(), patadas=(), pasos=(), guardia_baja=False, n=N, vis=1.0,
             alto_patada=220, seed=0):
    """Esqueleto sintetico (xy, vis). Los jabs son de la mano izquierda (la
    delantera en postura ortodoxa); las patadas, de la pierna derecha (trasera)."""
    rng = np.random.default_rng(seed)
    nombres = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
               "right_knee", "left_ankle", "right_ankle"]
    base = {"nose": (310, 70), "left_shoulder": (320, 100), "right_shoulder": (280, 100),
            "left_elbow": (330, 140), "right_elbow": (290, 140),
            "left_wrist": (340, 85), "right_wrist": (320, 85),
            "left_hip": (310, 200), "right_hip": (290, 200),
            "left_knee": (320, 290), "right_knee": (280, 290),
            "left_ankle": (340, 370), "right_ankle": (260, 370)}
    xy = {k: np.tile(np.array(v, float), (n, 1)) for k, v in base.items()}
    for j in jabs:
        xy["left_wrist"][:, 0] += bump(n, j, 16, 110)
        if guardia_baja:
            xy["right_wrist"][:, 1] += bump(n, j - 6, 30, 110)
    for k in patadas:
        xy["right_ankle"][:, 0] += bump(n, k, 24, 140)
        xy["right_ankle"][:, 1] -= bump(n, k, 24, alto_patada)
        xy["right_knee"][:, 1] -= bump(n, k, 24, alto_patada * 0.5)
    for p in pasos:  # paso de footwork: el pie se desplaza y apenas se levanta
        xy["left_ankle"][:, 0] += bump(n, p, 20, 90)
        xy["left_ankle"][:, 1] -= bump(n, p, 20, 15)
    for k in xy:
        xy[k] = xy[k] + rng.normal(0, 0.3, xy[k].shape)
    return xy, {k: np.full(n, vis) for k in nombres}


JABS = (60, 160, 260, 360, 460, 560)
PATADAS = (100, 300, 500)


# --------------------------------------------------------------------------
# Perfil
# --------------------------------------------------------------------------

def test_perfil_valida_rangos_y_valores():
    with pytest.raises(ValueError):
        cp.Perfil(altura_cm=50, peso_kg=70)
    with pytest.raises(ValueError):
        cp.Perfil(altura_cm=180, peso_kg=500)
    with pytest.raises(ValueError):
        cp.Perfil(altura_cm=180, peso_kg=70, postura="diestra")
    with pytest.raises(ValueError):
        cp.desde_dict({"altura_cm": 180, "peso_kg": 70, "talla": 42})


def test_perfil_niveles_y_lado_delantero():
    assert cp.Perfil(180, 70, experiencia_meses=3).nivel == "principiante"
    assert cp.Perfil(180, 70, experiencia_meses=6).nivel == "intermedio"
    assert cp.Perfil(180, 70, experiencia_meses=24).nivel == "avanzado"
    assert cp.Perfil(180, 70, postura="ortodoxa").lado_delantero == "left"
    assert cp.Perfil(180, 70, postura="zurda").lado_delantero == "right"


def test_perfil_escala_con_la_altura():
    a, b = cp.Perfil(160, 60), cp.Perfil(200, 60)
    assert b.cm_por_torso / a.cm_por_torso == pytest.approx(200 / 160)
    assert a.cm_por_torso == pytest.approx(0.288 * 160)


# --------------------------------------------------------------------------
# Deteccion
# --------------------------------------------------------------------------

def test_detecta_jabs_como_puño_delantero():
    xy, _ = luchador(jabs=JABS)
    golpes = cg.detectar(xy, FPS, "left")
    assert [g["tipo"] for g in golpes] == ["puño"] * len(JABS)
    assert all(g["lado"] == "left" and g["rol"] == "delantera" for g in golpes)
    for g, j in zip(golpes, JABS):
        assert abs(g["pico"] - (j + 8)) <= 4  # el pico de velocidad es a mitad del pulso


def test_postura_zurda_invierte_delantera_y_trasera():
    xy, _ = luchador(jabs=JABS)
    assert {g["rol"] for g in cg.detectar(xy, FPS, "right")} == {"trasera"}


def test_detecta_patadas_y_no_confunde_un_paso_con_una_patada():
    xy, _ = luchador(patadas=PATADAS, pasos=(200, 400, 600))
    golpes = cg.detectar(xy, FPS, "left")
    patadas = [g for g in golpes if g["tipo"] == "patada"]
    assert len(patadas) == len(PATADAS)
    assert all(g["lado"] == "right" and g["rol"] == "trasera" for g in patadas)
    assert not any(g["tipo"] == "patada" and g["lado"] == "left" for g in golpes)


def test_un_ajuste_de_guardia_sin_extension_no_es_un_puño():
    xy, _ = luchador()
    for j in JABS:  # la muneca sube y baja rapido pero no se aleja del hombro
        xy["left_wrist"][:, 1] -= bump(N, j, 10, 25)
    assert [g for g in cg.detectar(xy, FPS, "left") if g["tipo"] == "puño"] == []


def test_una_mano_que_cae_de_la_guardia_no_es_un_puño_salvo_en_vista_frontal():
    xy, _ = luchador()
    xy["right_wrist"][:, 1] += bump(N, 60, 30, 110)
    assert cg.detectar(xy, FPS, "left") == []
    # en vista frontal no se exige que la extension sea horizontal
    assert [g["lado"] for g in cg.detectar(xy, FPS, "left", solo_horizontal=False)] == ["right"]


def test_los_brazos_de_contrapeso_en_una_patada_no_son_puños():
    xy, _ = luchador(patadas=(300,))
    xy["left_wrist"][:, 0] += bump(N, 296, 16, 110)  # el brazo se abre durante la patada
    tipos = [g["tipo"] for g in cg.detectar(xy, FPS, "left")]
    assert tipos == ["patada"]


def test_dos_puños_simultaneos_de_manos_opuestas_son_uno_el_de_mas_extension():
    xy, _ = luchador(jabs=(200,))
    xy["right_wrist"][:, 0] += bump(N, 200, 16, 70)   # la otra mano tambien se mueve, menos
    golpes = [g for g in cg.detectar(xy, FPS, "left") if g["tipo"] == "puño"]
    assert [g["lado"] for g in golpes] == ["left"]


def test_golpes_con_el_extremo_invisible_se_descartan_al_detectar():
    xy, vis = luchador(jabs=JABS, patadas=PATADAS)
    for k in ("right_hip", "right_knee", "right_ankle"):
        vis[k][:] = 0.1
    golpes = cg.detectar(xy, FPS, "left", vis=vis)
    assert not any(g["tipo"] == "patada" for g in golpes)
    assert sum(g["tipo"] == "puño" for g in golpes) == len(JABS)


def test_el_reposo_de_los_pies_no_cuenta_como_levante():
    xy, vis = luchador(patadas=PATADAS)
    xy["right_ankle"][:, 1] -= 60   # un pie siempre mas alto en imagen (camara inclinada)
    xy["right_knee"][:, 1] -= 30
    filas, _ = medir(xy, vis)
    patadas = [f for f in filas if f["tipo"] == "patada"]
    assert len(patadas) == 3
    for f in patadas:
        assert f["altura_cm"] == pytest.approx(114, abs=6)       # la altura no se infla
        assert f["retorno_pierna_ms"] < 1000                      # y la pierna "vuelve"


def test_el_retorno_de_la_guardia_se_mide_desde_la_extension_maxima():
    # pulso de 16 frames: el brazo vuelve a su sitio unos 8 frames (~67 ms) despues del maximo
    xy, vis = luchador(jabs=JABS)
    filas, _ = medir(xy, vis)
    for f in filas:
        assert 30 < f["retorno_guardia_ms"] < 400, f["retorno_guardia_ms"]


def test_el_giro_de_cadera_sin_cadera_visible_no_da_numero():
    xy, vis = luchador(jabs=JABS)
    for k in ("left_hip", "right_hip"):
        vis[k][:] = 0.1
    assert all(np.isnan(f["giro_cadera_deg"]) for f in medir(xy, vis)[0])


def test_combinaciones_agrupa_por_separacion():
    g = [{"pico": p} for p in (0, 30, 60, 300, 330)]
    assert cg.combinaciones(g, FPS, 0.5) == [[0, 1, 2], [3, 4]]
    assert cg.combinaciones([], FPS) == []


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------

def medir(xy, vis, perfil=PERFIL):
    golpes = cg.detectar(xy, FPS, perfil.lado_delantero)
    return cm.medir(xy, vis, FPS, perfil, golpes)


def test_velocidad_en_ms_escala_con_la_altura_del_perfil():
    xy, vis = luchador(jabs=JABS)
    bajo = np.median([f["velocidad_ms"] for f in medir(xy, vis, cp.Perfil(160, 70))[0]])
    alto = np.median([f["velocidad_ms"] for f in medir(xy, vis, cp.Perfil(200, 70))[0]])
    assert alto / bajo == pytest.approx(200 / 160, rel=1e-6)
    # pico de velocidad de un pulso sin^2 de 110 px en 16 frames: pi/16 * 110 px/frame;
    # con el torso de 100 px = 0,288 * 160 cm
    esperado = np.pi / 16 * 1.10 * FPS * 0.288 * 160 / 100
    assert bajo == pytest.approx(esperado, rel=0.1)


def test_el_momento_usa_el_peso_del_perfil():
    xy, vis = luchador(jabs=JABS)
    a = medir(xy, vis, cp.Perfil(180, 60))[0][0]["momento_kgms"]
    b = medir(xy, vis, cp.Perfil(180, 90))[0][0]["momento_kgms"]
    assert b / a == pytest.approx(90 / 60)


def test_altura_de_patada_en_cm_y_zona():
    # levante de 220 px = 2,2 torsos; con 180 cm: 2,2 * 0,288 * 180 = 114 cm = 63 %
    xy, vis = luchador(patadas=PATADAS)
    filas, _ = medir(xy, vis)
    patadas = [f for f in filas if f["tipo"] == "patada"]
    assert len(patadas) == 3
    for f in patadas:
        assert f["altura_cm"] == pytest.approx(114, abs=6)
        assert f["zona"] == "media"
    xy, vis = luchador(patadas=PATADAS, alto_patada=110)
    assert {f["zona"] for f in medir(xy, vis)[0] if f["tipo"] == "patada"} == {"baja"}


def test_zona_patada_umbrales():
    assert cm.zona_patada(30) == "baja"
    assert cm.zona_patada(60) == "media"
    assert cm.zona_patada(85) == "alta"


def test_inclinacion_firmada_adelante_y_atras():
    cad, hom = (100, 200), (130, 100)
    assert cm.inclinacion_firmada(hom, cad, +1) > 0     # hombros por delante
    assert cm.inclinacion_firmada(hom, cad, -1) < 0     # golpeando hacia el otro lado
    assert cm.inclinacion_firmada((100, 100), cad, +1) == pytest.approx(0)


def test_guardia_baja_se_mide_en_el_puño():
    bien, vb = luchador(jabs=JABS)
    mal, vm = luchador(jabs=JABS, guardia_baja=True)
    g_bien = np.median([f["guardia_otra_mano_cm"] for f in medir(bien, vb)[0]])
    g_mal = np.median([f["guardia_otra_mano_cm"] for f in medir(mal, vm)[0]])
    assert g_mal > g_bien + 15  # cm


def test_porcentaje_de_guardia_arriba_de_la_sesion():
    bien, vb = luchador(jabs=JABS)
    mal, vm = luchador(jabs=JABS, guardia_baja=True)
    assert medir(bien, vb)[1]["guardia_arriba_pct"] > 95
    assert medir(mal, vm)[1]["guardia_arriba_pct"] < medir(bien, vb)[1]["guardia_arriba_pct"] - 10


def test_sin_visibilidad_el_golpe_no_es_fiable_y_no_da_numeros():
    xy, vis = luchador(jabs=JABS, vis=0.2)
    filas, sesion = medir(xy, vis)
    assert filas and not any(f["fiable"] for f in filas)
    assert all(np.isnan(f["velocidad_ms"]) for f in filas)
    assert sesion["fiables"] == 0


# --------------------------------------------------------------------------
# Correcciones
# --------------------------------------------------------------------------

def filas_con(metrica, valores, tipo="puño"):
    return [{"tipo": tipo, "fiable": True, "t_s": float(i), metrica: v}
            for i, v in enumerate(valores)]


def test_exceso_es_cero_en_ok_y_uno_en_mal():
    assert cc._exceso(10, 10, 20, cc.ALTO) == 0
    assert cc._exceso(20, 10, 20, cc.ALTO) == 1
    assert cc._exceso(6, 15, 6, cc.BAJO) == 1
    assert cc.estado(-0.1) == cc.BIEN and cc.estado(0.5) == cc.MEJORAR \
        and cc.estado(1.2) == cc.PRIORITARIO


def regla(id_):
    return next(r for r in cc.REGLAS if r["id"] == id_)


def test_el_mismo_golpe_se_juzga_distinto_segun_el_nivel():
    # retorno de 500 ms: bien para un principiante (<=450 es "bien", 500 "a mejorar"),
    # prioritario para un avanzado (>500 es prioritario)
    filas = filas_con("retorno_guardia_ms", [520, 540, 560])
    nov = cc.evaluar_regla(regla("retorno_puño"), filas, {}, cp.Perfil(180, 70, 2))
    ava = cc.evaluar_regla(regla("retorno_puño"), filas, {}, cp.Perfil(180, 70, 36))
    assert nov["estado"] == cc.MEJORAR
    assert ava["estado"] == cc.PRIORITARIO
    assert ava["gravedad"] > nov["gravedad"]


def test_los_umbrales_en_cm_dependen_de_la_altura():
    # la misma guardia de 0,8 torsos bajo la nariz son mas cm en una persona alta
    r = regla("guardia_puño")
    bajo = cc.evaluar_regla(r, filas_con("guardia_otra_mano_cm", [30] * 3), {}, cp.Perfil(160, 70, 12))
    alto = cc.evaluar_regla(r, filas_con("guardia_otra_mano_cm", [30] * 3), {}, cp.Perfil(200, 70, 12))
    assert alto["umbral_ok"] / bajo["umbral_ok"] == pytest.approx(200 / 160)
    assert bajo["estado"] != alto["estado"] or bajo["exceso"] > alto["exceso"]


def test_pocos_golpes_dan_sin_dato():
    r = cc.evaluar_regla(regla("guardia_puño"), filas_con("guardia_otra_mano_cm", [90, 90]), {}, PERFIL)
    assert r["estado"] == "sin dato" and r["n"] == 2


def test_el_giro_de_cadera_solo_se_evalua_de_lado():
    filas = filas_con("giro_cadera_deg", [2, 3, 2])
    assert cc.evaluar_regla(regla("giro_cadera_puño"), filas, {}, PERFIL, "frontal")["estado"] == "sin dato"
    assert cc.evaluar_regla(regla("giro_cadera_puño"), filas, {}, PERFIL, "lateral")["estado"] == cc.PRIORITARIO


def test_los_golpes_no_fiables_no_cuentan():
    filas = filas_con("guardia_otra_mano_cm", [90, 90, 90])
    for f in filas:
        f["fiable"] = False
    assert cc.evaluar_regla(regla("guardia_puño"), filas, {}, PERFIL)["estado"] == "sin dato"


def test_la_peor_pega_apunta_al_golpe_mas_claro():
    filas = filas_con("guardia_otra_mano_cm", [10, 80, 12, 11])
    r = cc.evaluar_regla(regla("guardia_puño"), filas, {}, PERFIL)
    assert r["peor_t_s"] == 1.0


def test_notas_del_perfil_flexibilidad_y_limitaciones():
    p = cp.Perfil(180, 70, 2, flexibilidad="baja", limitaciones=["rodilla derecha"])
    notas = " ".join(cc.notas_perfil(p, True))
    assert "UNA corrección" in notas and "flexibilidad baja" in notas and "rodilla derecha" in notas
    assert "flexibilidad" not in " ".join(cc.notas_perfil(p, False))


# --------------------------------------------------------------------------
# De punta a punta
# --------------------------------------------------------------------------

def test_guardia_baja_sale_la_primera_en_la_lista():
    xy, vis = luchador(jabs=JABS, guardia_baja=True)
    r = ac.analizar_xy(xy, vis, FPS, PERFIL)
    ordenadas = [x for x in r["reglas"] if "gravedad" in x]
    assert ordenadas[0]["id"] == "guardia_puño"
    assert ordenadas[0]["estado"] in (cc.PRIORITARIO, cc.MEJORAR)


def test_el_informe_se_escribe_con_las_secciones_y_el_aviso():
    xy, vis = luchador(jabs=JABS, patadas=PATADAS, guardia_baja=True)
    r = ac.analizar_xy(xy, vis, FPS, cp.Perfil(180, 75, 3, flexibilidad="baja",
                                                limitaciones=["hombro izquierdo"]))
    md = ac.informe_markdown(r, "mi_clip.mp4")
    for s in ("Qué corregir, por orden", "Altura de las patadas", "Cómo se ha medido",
              "hombro izquierdo", "no están calibrados", "mi_clip.mp4"):
        assert s in md


def test_un_clip_sin_golpes_no_rompe():
    xy, vis = luchador()
    r = ac.analizar_xy(xy, vis, FPS, PERFIL)
    assert r["golpes"] == [] and r["patadas"] is None
    assert r["calidad"] == ["no se ha detectado ningún golpe"]
    assert "No se puede decir qué corregir" in ac.informe_markdown(r)


def test_un_clip_con_el_cuerpo_cortado_avisa_en_vez_de_dar_el_visto_bueno():
    xy, vis = luchador(jabs=JABS, patadas=PATADAS)
    for k in ("left_elbow", "left_wrist"):
        vis[k][:] = 0.1                      # el brazo adelantado no se ve
    r = ac.analizar_xy(xy, vis, FPS, PERFIL)
    assert r["calidad"], "debe haber motivo de calidad"
    md = ac.informe_markdown(r)
    assert "no da para conclusiones" in md and "cuerpo entero en plano" in md


def test_un_clip_bueno_no_lleva_aviso_de_calidad():
    xy, vis = luchador(jabs=JABS, patadas=PATADAS)
    assert ac.analizar_xy(xy, vis, FPS, PERFIL)["calidad"] == []


def test_la_vista_frontal_avisa_de_que_puños_y_patadas_se_detectan_mal():
    xy, vis = luchador(jabs=JABS)
    assert any("Vista frontal" in n for n in ac.analizar_xy(xy, vis, FPS, PERFIL, "frontal")["notas"])
    assert not any("Vista frontal" in n for n in ac.analizar_xy(xy, vis, FPS, PERFIL, "lateral")["notas"])


def test_en_frontal_no_se_presentan_pasos_o_patadas_como_patadas_fiables():
    xy, vis = luchador(patadas=PATADAS)
    frontal = ac.analizar_xy(xy, vis, FPS, PERFIL, "frontal")
    lateral = ac.analizar_xy(xy, vis, FPS, PERFIL, "lateral")
    assert frontal["sesion"]["patadas"] == 0
    assert frontal["patadas"] is None
    assert lateral["sesion"]["patadas"] > 0


# --------------------------------------------------------------------------
# Referencias de profesionales
# --------------------------------------------------------------------------

def clip_pro(guardia_cm, perfil=PERFIL, n=4):
    filas = [{"tipo": "puño", "fiable": True, "guardia_otra_mano_cm": guardia_cm,
              "retorno_guardia_ms": 300.0} for _ in range(n)]
    return cr.medianas_clip(filas, perfil)


def test_las_medidas_en_cm_se_guardan_en_torsos_y_valen_para_otra_altura():
    bajo, alto = cp.Perfil(160, 60), cp.Perfil(200, 90)
    en_torsos = cr.a_unidad_comun("guardia_otra_mano_cm", 0.3 * bajo.cm_por_torso, bajo)
    assert en_torsos == pytest.approx(0.3)
    assert cr.de_unidad_comun("guardia_otra_mano_cm", en_torsos, alto) == pytest.approx(0.3 * alto.cm_por_torso)
    # los ms no dependen del cuerpo
    assert cr.a_unidad_comun("retorno_guardia_ms", 300, alto) == 300


def test_el_rango_necesita_minimo_de_clips_y_se_ensancha_con_margen():
    c = [clip_pro(g) for g in (10, 20)]
    assert cr.construir(dict(zip("ab", c))) == {}
    clips = {n: clip_pro(g) for n, g in zip("abc", (10, 20, 30))}
    ref = cr.construir(clips)
    lo, hi, n = cr.rango(ref, "puño", "guardia_otra_mano_cm", PERFIL)
    ancho = (30 - 10)
    assert n == 3 and lo == pytest.approx(10 - cr.MARGEN * ancho) and hi == pytest.approx(30 + cr.MARGEN * ancho)
    assert cr.posicion(40, (lo, hi, n)) == "por encima" and cr.posicion(20, (lo, hi, n)) == "dentro" \
        and cr.posicion(0, (lo, hi, n)) == "por debajo"


def test_un_clip_con_pocos_golpes_no_aporta_referencia():
    filas = [{"tipo": "puño", "fiable": True, "guardia_otra_mano_cm": 20.0}] * 2
    assert cr.medianas_clip(filas, PERFIL) == {}


def test_la_evaluacion_incluye_el_rango_profesional_y_el_informe_lo_dice():
    clips = {n: clip_pro(g) for n, g in zip("abc", (10, 20, 30))}
    ref = cr.construir(clips)
    xy, vis = luchador(jabs=JABS, guardia_baja=True)
    r = ac.analizar_xy(xy, vis, FPS, PERFIL, "lateral", None, ref)
    g = next(x for x in r["reglas"] if x["id"] == "guardia_puño")
    assert g["referencia"]["posicion"] == "por encima" and g["referencia"]["n_clips"] == 3
    assert "Profesionales de referencia" in ac.informe_markdown(r)
    assert "referencia" not in next(x for x in ac.analizar_xy(xy, vis, FPS, PERFIL)["reglas"]
                                    if x["id"] == "guardia_puño")


# --------------------------------------------------------------------------
# Tipo de puño
# --------------------------------------------------------------------------

def test_clasificar_puño_recto_gancho_y_uppercut():
    cl = cm.clasificar_puño
    assert cl(170, 0.1, 1.0, "delantera", "lateral") == ("recto", "jab")
    assert cl(165, 0.1, 1.0, "trasera", "lateral") == ("recto", "cruzado")
    assert cl(100, 0.1, 0.8, "delantera", "lateral") == ("gancho", "gancho adelantado")
    assert cl(95, 0.6, 0.2, "trasera", "diagonal") == ("uppercut", "uppercut trasero")
    # sube pero avanza mas: sigue siendo gancho
    assert cl(95, 0.4, 0.9, "trasera", "lateral")[0] == "gancho"


def test_en_vista_frontal_y_sin_codo_no_se_clasifica():
    assert cm.clasificar_puño(170, 0.1, 1.0, "delantera", "frontal") == (None, None)
    assert cm.clasificar_puño(float("nan"), 0.1, 1.0, "delantera", "lateral") == (None, None)


def test_el_jab_sintetico_sale_como_jab_y_el_informe_cuenta_los_tipos():
    xy, vis = luchador(jabs=JABS)
    for j in JABS:  # el codo acompana: brazo estirado en linea hombro-muneca
        t = np.arange(N)
        pulso = bump(N, j, 16, 1.0)
        xy["left_elbow"][:, 0] = xy["left_elbow"][:, 0] + pulso * 55
        xy["left_elbow"][:, 1] = xy["left_elbow"][:, 1] - pulso * 52
    r = ac.analizar_xy(xy, vis, FPS, PERFIL)
    nombres = [g["nombre"] for g in r["golpes"] if g["tipo"] == "puño"]
    assert nombres and set(nombres) == {"jab"}
    assert r["sesion"]["tipos_de_puño"] == {"jab": len(JABS)}
    assert "6 jab" in ac.informe_markdown(r)
