"""Que corregir primero, segun el nivel, el cuerpo y las limitaciones de la persona.

Para cada regla (una metrica de puño, de patada o de la sesion) se toma la
mediana de los golpes fiables y se compara con dos umbrales que dependen del
nivel del perfil: `ok` (hasta aqui bien) y `mal` (a partir de aqui, prioritario).
Entre medias, "a mejorar". La gravedad de la regla es cuanto se pasa del
umbral, normalizado, por el peso de la regla. Se devuelven ordenadas: lo
primero de la lista es lo que mas compensa corregir.

AVISO: los umbrales y los diagnosticos son orientativos. Salen de metodologia
general de boxeo y K-1 redactada para este proyecto, NO estan calibrados con
clips de referencia de profesionales ni validados por un entrenador. Son
hipotesis para mirar en el video, no un veredicto; y lo que diga tu entrenador
pesa mas que cualquier cosa de aqui. Si un entrenador los revisa, este es el
unico fichero que hay que tocar.

Las unidades van en cm (con la altura del perfil) donde la metrica es de
distancia, para que el texto hable de tu cuerpo.
"""

import numpy as np

import combate_referencias as cr

MIN_GOLPES = 3        # golpes fiables para dar un valor de sesion
PRIMERAS = 3          # cuantas correcciones se destacan
BIEN, MEJORAR, PRIORITARIO = "bien", "a mejorar", "prioritario"
ALTO, BAJO = "alto", "bajo"   # sentido en que salirse del umbral es un defecto

# umbrales[nivel] = (ok, mal). En sentido ALTO: valor <= ok es bien, > mal es
# prioritario. En sentido BAJO, al reves.
REGLAS = [
    {
        "id": "guardia_puño", "tipo": "puño", "metrica": "guardia_otra_mano_cm",
        "sentido": ALTO, "peso": 1.0, "unidad": "cm bajo la nariz",
        "umbrales": {"principiante": (0.55, 0.90), "intermedio": (0.50, 0.80),
                     "avanzado": (0.45, 0.70)},
        "en_torsos": True,
        "problema": "la mano que no golpea baja de la guardia al lanzar el puño",
        "causas": ["se atiende solo a la mano que pega y la otra se olvida",
                   "el hombro de la mano que golpea tira del brazo contrario",
                   "cansancio: la guardia cae al final de las series"],
        "que_mirar": "en el video, la mano trasera durante el golpe: debe seguir "
                     "pegada a la barbilla o la mejilla, no a la altura del pecho.",
        "ejercicios": ["sombra frente a un espejo tocándote la barbilla con la mano "
                       "libre después de cada golpe",
                       "jab-cruzado lento (3 segundos por golpe) vigilando la mano que no pega",
                       "rondas de sombra de 1 minuto solo con guardia, sin golpear"],
    },
    {
        "id": "retorno_puño", "tipo": "puño", "metrica": "retorno_guardia_ms",
        "sentido": ALTO, "peso": 0.8, "unidad": "ms",
        "umbrales": {"principiante": (450, 800), "intermedio": (350, 600),
                     "avanzado": (300, 500)},
        "problema": "el puño tarda en volver a la guardia después del golpe",
        "causas": ["el brazo se queda estirado o relajado tras el impacto",
                   "se golpea con tensión y se tarda en soltar",
                   "se mira el golpe en vez de pensar ya en el siguiente"],
        "que_mirar": "cuántos fotogramas pasan entre la máxima extensión y la muñeca "
                     "de vuelta junto a la cara; debería ser un movimiento de ida y vuelta.",
        "ejercicios": ["jab y vuelta: lanzar y recoger el puño con el mismo ritmo, "
                       "contando '1-y' en voz alta",
                       "sombra con goma elástica ligera en la muñeca para sentir la vuelta",
                       "golpear al saco soltando el brazo tras cada impacto, sin empujar"],
    },
    {
        "id": "inclinacion_puño", "tipo": "puño", "metrica": "inclinacion_deg",
        "sentido": ALTO, "peso": 0.5, "unidad": "grados hacia delante",
        "umbrales": {"principiante": (18, 30), "intermedio": (15, 25),
                     "avanzado": (12, 20)},
        "problema": "el tronco se va hacia delante al golpear",
        "causas": ["se llega al objetivo con el cuerpo en vez de con el brazo",
                   "distancia mal medida: se golpea desde demasiado lejos"],
        "que_mirar": "el ángulo de los hombros sobre las caderas en el momento de máxima "
                     "extensión: si se pasan de la rodilla delantera, hay desequilibrio.",
        "ejercicios": ["golpear a la palma de un compañero a distancia ajustada, "
                       "sin dar pasos hacia delante",
                       "sombra con un paso atrás tras cada serie de dos golpes"],
    },
    {
        "id": "giro_cadera_puño", "tipo": "puño", "metrica": "giro_cadera_deg",
        "sentido": BAJO, "peso": 0.4, "unidad": "grados de giro",
        "umbrales": {"principiante": (15, 6), "intermedio": (20, 10),
                     "avanzado": (25, 14)},
        "solo_vista": ("lateral", "diagonal"),
        "problema": "las caderas giran poco al golpear: el puño es solo de brazo",
        "causas": ["los pies no pivotan; el golpe sale sin transferir peso",
                   "pies muy abiertos o muy juntos"],
        "que_mirar": "las caderas vistas desde arriba o de lado: deben acompañar al hombro, "
                     "con el talón trasero girando en los golpes de mano trasera.",
        "ejercicios": ["cruzado en sombra pivotando el pie trasero, sin soltar la guardia",
                       "giros de cadera con el pie trasero sobre un papel para deslizar"],
    },
    {
        "id": "guardia_patada", "tipo": "patada", "metrica": "guardia_en_patada_cm",
        "sentido": ALTO, "peso": 1.0, "unidad": "cm bajo la nariz",
        "umbrales": {"principiante": (0.65, 1.00), "intermedio": (0.60, 0.90),
                     "avanzado": (0.55, 0.80)},
        "en_torsos": True,
        "problema": "las manos bajan al lanzar la patada",
        "causas": ["se usan los brazos para equilibrar",
                   "falta de equilibrio sobre la pierna de apoyo"],
        "que_mirar": "las dos manos en el pico de la patada: deben seguir arriba para "
                     "protegerte del contragolpe mientras estás a una pierna.",
        "ejercicios": ["patadas de sombra manteniendo los puños a la altura de las mejillas "
                       "de principio a fin",
                       "patada al paddle del compañero con guardia vigilada por él"],
    },
    {
        "id": "retorno_patada", "tipo": "patada", "metrica": "retorno_pierna_ms",
        "sentido": ALTO, "peso": 0.7, "unidad": "ms",
        "umbrales": {"principiante": (700, 1100), "intermedio": (550, 850),
                     "avanzado": (450, 700)},
        "problema": "la pierna tarda en volver al suelo y recoger la guardia",
        "causas": ["se deja caer la pierna en vez de recogerla",
                   "falta de fuerza o de control de la cadera",
                   "se pierde el equilibrio y hay que reajustarse"],
        "que_mirar": "cuántos fotogramas pasan entre la máxima altura del pie y la pierna "
                     "recogida: en un contraataque, esa pierna en el aire es el riesgo.",
        "ejercicios": ["patada y recogida (la pierna vuelve por el mismo camino) al ritmo "
                       "'1-y', sin apoyar entre repeticiones",
                       "patadas con tobillera elástica para trabajar la vuelta",
                       "equilibrio a una pierna 30 segundos por lado, con los ojos abiertos"],
    },
    {
        "id": "inclinacion_patada", "tipo": "patada", "metrica": "inclinacion_atras_deg",
        "sentido": ALTO, "peso": 0.6, "unidad": "grados hacia atrás",
        "umbrales": {"principiante": (25, 40), "intermedio": (22, 35),
                     "avanzado": (20, 30)},
        "problema": "el tronco se echa hacia atrás al patear",
        "causas": ["se levanta la pierna con el tronco en lugar de con la cadera",
                   "movilidad insuficiente en la cadera para subir la pierna",
                   "contrapeso para no caerse al llegar alto"],
        "que_mirar": "el ángulo del tronco en el punto más alto de la patada; si sube "
                     "demasiado echándote atrás, puede que la altura no sea realista "
                     "todavía y convenga bajar el objetivo.",
        "ejercicios": ["patadas bajas y medias con técnica limpia antes de subir de altura",
                       "movilidad de cadera: apertura 90/90 y estiramiento del psoas",
                       "patada contra un saco colgado a la altura de tu cadera, tronco vertical"],
    },
    {
        "id": "guardia_sesion", "tipo": "sesion", "metrica": "guardia_arriba_pct",
        "sentido": BAJO, "peso": 0.9, "unidad": "% del tiempo",
        "umbrales": {"principiante": (60, 40), "intermedio": (70, 50),
                     "avanzado": (80, 60)},
        "problema": "las manos pasan demasiado tiempo fuera de guardia",
        "causas": ["la guardia se relaja entre combinaciones, sobre todo al moverse",
                   "cansancio a lo largo de la ronda"],
        "que_mirar": "las manos mientras te mueves o descansas entre golpes: es cuando "
                     "más se olvida la guardia.",
        "ejercicios": ["rondas de sombra de 3 minutos moviéndote con guardia, contando cuántas "
                       "veces la bajas",
                       "trabajo de footwork con las manos pegadas a la cara"],
    },
]


def _valores(filas, regla):
    """Pares (valor, golpe) de los golpes fiables del tipo de la regla."""
    vals = [(f[regla["metrica"]], f) for f in filas
            if f["tipo"] == regla["tipo"] and f.get("fiable")
            and f.get(regla["metrica"]) is not None and not np.isnan(f[regla["metrica"]])]
    return vals


def _exceso(valor, ok, mal, sentido):
    """0 = en el umbral ok (o mejor); 1 = en el umbral mal; >1 = peor. Funcion pura."""
    if sentido == ALTO:
        return (valor - ok) / (mal - ok)
    return (ok - valor) / (ok - mal)


def estado(exceso):
    if exceso <= 0:
        return BIEN
    return MEJORAR if exceso <= 1 else PRIORITARIO


def evaluar_regla(regla, filas, sesion, perfil, vista="lateral"):
    ok, mal = regla["umbrales"][perfil.nivel]
    cm = perfil.cm_por_torso
    escala = cm if regla.get("en_torsos") else 1.0
    res = {"id": regla["id"], "metrica": regla["metrica"], "tipo": regla["tipo"],
           "unidad": regla["unidad"], "sentido": regla["sentido"],
           "problema": regla["problema"],
           "umbral_ok": ok * escala, "umbral_mal": mal * escala}
    solo = regla.get("solo_vista")
    if solo and vista not in solo:
        return {**res, "estado": "sin dato", "motivo": f"solo se mide en vista {'/'.join(solo)}"}

    if regla["tipo"] == "sesion":
        valor = sesion.get(regla["metrica"])
        n = 1
        peor = None
        if valor is None or np.isnan(valor):
            return {**res, "estado": "sin dato", "motivo": "no hay frames válidos"}
        valores = [valor]
    else:
        pares = _valores(filas, regla)
        n = len(pares)
        if n < MIN_GOLPES:
            return {**res, "estado": "sin dato", "n": n,
                    "motivo": f"solo {n} golpe(s) fiable(s); hacen falta {MIN_GOLPES}"}
        valores = [v for v, _ in pares]
        valor = float(np.median(valores))
        elegir = max if regla["sentido"] == ALTO else min
        peor = elegir(pares, key=lambda p: p[0])[1]

    exceso = _exceso(valor, ok * escala, mal * escala, regla["sentido"])
    fuera = sum(1 for v in valores if _exceso(v, ok * escala, mal * escala, regla["sentido"]) > 0)
    res.update({"valor": float(valor), "estado": estado(exceso), "exceso": float(exceso),
                "n": n, "golpes_fuera": int(fuera),
                "gravedad": float(np.clip(exceso, 0, 1.5) * regla["peso"])})
    if peor is not None:
        res["peor_t_s"] = peor["t_s"]
    return res


def altura_patadas(filas, perfil):
    """Resumen informativo de la altura de las patadas (no es un defecto: la
    zona que se busca la decide quien patea y su entrenador)."""
    pat = [f for f in filas if f["tipo"] == "patada" and f.get("fiable")
           and not np.isnan(f["altura_pct"])]
    if not pat:
        return None
    res = {"n": len(pat), "altura_max_cm": float(max(f["altura_cm"] for f in pat)),
           "altura_mediana_cm": float(np.median([f["altura_cm"] for f in pat])),
           "zonas": {z: sum(f["zona"] == z for f in pat) for z in ("baja", "media", "alta")},
           # alturas de referencia para ESTA persona (Winter, fraccion de la altura)
           "referencia_cm": {"rodilla": round(0.285 * perfil.altura_cm),
                             "cadera": round(0.530 * perfil.altura_cm),
                             "pecho": round(0.720 * perfil.altura_cm),
                             "cabeza": round(0.870 * perfil.altura_cm)}}
    return res


def evaluar(filas, sesion, perfil, vista="lateral", ref=None):
    """Todas las reglas evaluadas y ordenadas por gravedad (las sin dato al final).
    Con `ref` (combate_referencias.construir) cada regla con valor lleva ademas
    el rango de los profesionales y si la persona esta dentro, por encima o por
    debajo."""
    resultados = []
    for r in REGLAS:
        res = evaluar_regla(r, filas, sesion, perfil, vista)
        rg = cr.rango(ref, r["tipo"], r["metrica"], perfil) if "valor" in res else None
        if rg:
            res["referencia"] = {"min": rg[0], "max": rg[1], "n_clips": rg[2],
                                 "posicion": cr.posicion(res["valor"], rg)}
        res["que_mirar"] = r["que_mirar"]
        res["causas"] = r["causas"]
        res["ejercicios"] = r["ejercicios"]
        resultados.append(res)
    return sorted(resultados, key=lambda r: -r.get("gravedad", -1))


def notas_perfil(perfil, patadas):
    """Avisos que salen del perfil declarado, no de las metricas."""
    notas = []
    if perfil.nivel == "principiante":
        notas.append("Llevas poco tiempo practicando: céntrate en UNA corrección cada "
                     "semana, la primera de la lista, antes de pasar a la siguiente.")
    if perfil.flexibilidad == "baja" and patadas:
        notas.append("Declaras flexibilidad baja: no persigas la altura de la patada. "
                     "Una patada media limpia vale más que una alta forzada; trabaja la "
                     "movilidad de cadera aparte.")
    for lim in perfil.limitaciones:
        notas.append(f"Limitación declarada: «{lim}». Los ejercicios son generales: "
                     "adáptalos con tu entrenador o evita los que te carguen esa zona.")
    return notas
