"""Que significa salirse del rango de los profesionales en cada metrica, y que
hacer al respecto.

Dos piezas:

- SENTIDO_PROBLEMA: en que direccion salirse del rango suele ser un defecto en
  un aficionado. Salirse por el otro lado no se oculta, pero se valora "sin
  problema": en la validacion, Alcaraz gira mas y termina mas alto que el
  resto de profesionales, y eso no es algo que corregir.
- DIAGNOSTICOS: para cada metrica y sentido problematico, el defecto que
  suele haber detras, sus causas habituales, que mirar en el video para
  confirmarlo y ejercicios.

AVISO: el contenido de DIAGNOSTICOS es metodologia general de ensenanza del
tenis, redactada para este proyecto. No esta validado por un entrenador ni
contra datos: es una lista de hipotesis razonables para cada numero, no un
diagnostico. El informe lo presenta asi. Si un entrenador lo revisa, este es
el unico fichero que hay que tocar.
"""

ABAJO, ARRIBA, AMBOS = "abajo", "arriba", "ambos"

SENTIDO_PROBLEMA = {
    "rodilla_min": ARRIBA,      # angulo alto = pierna poco flexionada
    "tronco_impacto": ARRIBA,   # muy inclinado = desequilibrio
    "ancho_apoyos": AMBOS,
    "altura_impacto": AMBOS,
    "codo_impacto": ABAJO,      # angulo bajo = brazo encogido
    "impacto_delante": ABAJO,   # negativo = tarde
    "terminacion": ABAJO,
    "giro_hombros": ABAJO,
    "peso_delante": ABAJO,
    "alcance": ABAJO,
}

FONDO = ("derecha", "reves_2manos")
DERECHA = ("derecha",)
SAQUE = ("saque",)

DIAGNOSTICOS = [
    {
        "metrica": "giro_hombros", "sentido": ABAJO, "golpes": FONDO,
        "problema": "golpe de brazo: el tronco gira poco",
        "causas": ["preparación tardía o incompleta: los hombros no llegan a ponerse "
                   "de lado antes de golpear",
                   "golpear con los pies y el pecho mirando a la red todo el gesto"],
        "que_mirar": "en la preparación, desde el lateral deberías ver casi toda tu "
                     "espalda; después del golpe, el pecho mirando hacia donde mandas la bola",
        "ejercicios": ["sombras: girar los hombros hasta que el hombro contrario apunte a "
                       "la bola, pausa, y golpear dejando que el giro arrastre al brazo",
                       "preparar con la mano libre en el cuello de la raqueta y no soltarla "
                       "hasta que los hombros estén de lado",
                       "lanzamientos laterales con un balón medicinal ligero, girando el "
                       "tronco con los brazos casi quietos"],
    },
    {
        "metrica": "terminacion", "sentido": ABAJO, "golpes": DERECHA,
        "problema": "terminación corta: el brazo se frena después del impacto",
        "causas": ["golpe empujado y plano en vez de de abajo arriba",
                   "frenar el brazo por miedo a que la bola se vaya larga"],
        "que_mirar": "donde acaba la raqueta: deberia pasar por encima del hombro "
                     "contrario, no quedarse a la altura del pecho",
        "ejercicios": ["terminar cada golpe recogiendo la raqueta con la mano libre por "
                       "encima del hombro contrario",
                       "peloteo pasando la bola por encima de una cuerda tendida un metro "
                       "por encima de la red, para obligar a golpear de abajo arriba"],
    },
    {
        "metrica": "terminacion", "sentido": ABAJO, "golpes": ("reves_2manos",),
        "problema": "terminación corta: los brazos se frenan después del impacto",
        "causas": ["golpe empujado y plano en vez de de abajo arriba"],
        "que_mirar": "donde acaban las manos: deberian terminar por encima del hombro "
                     "de la mano dominante",
        "ejercicios": ["sombras terminando con las dos manos por encima del hombro y los "
                       "codos apuntando hacia donde va la bola"],
    },
    {
        "metrica": "peso_delante", "sentido": ABAJO, "golpes": FONDO,
        "problema": "peso atrás en el impacto",
        "causas": ["llegar tarde a la bola y golpear retrocediendo",
                   "colocarse demasiado cerca del bote"],
        "que_mirar": "si después del golpe te vas hacia atrás o tienes que dar un paso "
                     "atrás para no perder el equilibrio",
        "ejercicios": ["con bola dada, golpear y aguantar dos segundos en equilibrio sobre "
                       "la pierna que queda delante",
                       "colocarse un paso más atrás del bote para poder ir hacia la bola"],
    },
    {
        "metrica": "rodilla_min", "sentido": ARRIBA, "golpes": FONDO,
        "problema": "piernas rígidas: flexionas poco antes de golpear",
        "causas": ["golpear solo con el tren superior",
                   "colocarse con pasos largos en vez de ajustar con pasos cortos"],
        "que_mirar": "si en la preparación bajas la cadera o te quedas a la misma altura "
                     "que de pie",
        "ejercicios": ["sombras bajando como para sentarte en la preparación y empujando "
                       "hacia arriba al golpear",
                       "peloteo con bolas bajas que obliguen a flexionar"],
    },
    {
        "metrica": "rodilla_min", "sentido": ARRIBA, "golpes": SAQUE,
        "problema": "poca flexión de piernas en el saque",
        "causas": ["sacar solo con el brazo, sin cargar las piernas durante el lanzamiento"],
        "que_mirar": "si las rodillas se doblan mientras sube la bola",
        "ejercicios": ["sombras de saque marcando la flexión de rodillas al soltar la bola "
                       "y la extensión al golpear"],
    },
    {
        "metrica": "tronco_impacto", "sentido": ARRIBA, "golpes": FONDO,
        "problema": "tronco muy inclinado en el impacto",
        "causas": ["llegar mal colocado y compensar inclinándose en vez de con los pies",
                   "inclinarse para llegar a bolas bajas en vez de flexionar las piernas"],
        "que_mirar": "si la cabeza se va hacia un lado durante el golpe",
        "ejercicios": ["golpear y mantener la postura un segundo con la cabeza quieta",
                       "pasos cortos de ajuste antes de cada golpe"],
    },
    {
        "metrica": "ancho_apoyos", "sentido": ABAJO, "golpes": FONDO,
        "problema": "apoyos estrechos en el impacto",
        "causas": ["golpear sin haber terminado de colocar los pies"],
        "que_mirar": "la distancia entre los pies al golpear: algo más que el ancho de "
                     "los hombros",
        "ejercicios": ["sombras con dos marcas en el suelo algo más separadas que los "
                       "hombros, golpeando con un pie en cada una"],
    },
    {
        "metrica": "ancho_apoyos", "sentido": ARRIBA, "golpes": FONDO,
        "problema": "apoyos muy abiertos en el impacto",
        "causas": ["llegar estirado a la bola con un último paso largo"],
        "que_mirar": "si después del golpe te cuesta recuperar la posición",
        "ejercicios": ["ajustar con pasos cortos al final para llegar a la bola con la "
                       "base bajo el cuerpo"],
    },
    {
        "metrica": "altura_impacto", "sentido": ABAJO, "golpes": FONDO,
        "problema": "impacto bajo: golpeas la bola cuando ya cae",
        "causas": ["colocarse demasiado lejos del bote"],
        "que_mirar": "a que altura de tu cuerpo esta la bola al golpear",
        "ejercicios": ["acercarse al bote para golpear la bola entre la cintura y el pecho"],
    },
    {
        "metrica": "altura_impacto", "sentido": ARRIBA, "golpes": FONDO,
        "problema": "impacto alto",
        "causas": ["colocarse demasiado cerca del bote con bolas que suben"],
        "que_mirar": "a que altura de tu cuerpo esta la bola al golpear",
        "ejercicios": ["retroceder un paso con bolas altas para dejarlas bajar hasta la "
                       "cintura o el pecho"],
    },
    {
        "metrica": "codo_impacto", "sentido": ABAJO, "golpes": FONDO,
        "problema": "brazo encogido: la bola te pilla pegada al cuerpo",
        "causas": ["colocarse demasiado cerca de la bola"],
        "que_mirar": "el espacio entre tu cuerpo y la bola en el impacto",
        "ejercicios": ["con bola dada, golpear con una marca en el suelo a un brazo y una "
                       "raqueta de distancia del pie"],
    },
    {
        "metrica": "impacto_delante", "sentido": ABAJO, "golpes": FONDO,
        "problema": "impacto tarde: golpeas la bola a la altura del cuerpo o detras",
        "causas": ["preparación tardía", "no ir hacia la bola"],
        "que_mirar": "si en el impacto la bola esta por delante del pie delantero",
        "ejercicios": ["empezar la preparación cuando la bola cruza la red",
                       "atrapar la bola con la mano por delante del pie delantero, y luego "
                       "golpear en ese mismo punto"],
    },
    {
        "metrica": "alcance", "sentido": ABAJO, "golpes": SAQUE,
        "problema": "impacto bajo en el saque",
        "causas": ["lanzamiento de bola bajo", "no extender el brazo ni el cuerpo hacia arriba"],
        "que_mirar": "si en el impacto el brazo esta estirado del todo",
        "ejercicios": ["lanzamientos sin golpear dejando caer la bola: deberia subir algo "
                       "más alto que la raqueta con el brazo estirado"],
    },
]


def valoracion(metrica, estado):
    """'a revisar', 'sin problema' o None (si no se comparo). Funcion pura."""
    if estado == "dentro del rango":
        return "sin problema"
    sentido = {"por debajo del rango": ABAJO, "por encima del rango": ARRIBA}.get(estado)
    if sentido is None:
        return None
    problema = SENTIDO_PROBLEMA.get(metrica, AMBOS)
    return "a revisar" if problema in (AMBOS, sentido) else "sin problema"


def diagnostico(metrica, estado, golpe):
    """Entrada de DIAGNOSTICOS para una metrica fuera del rango en su sentido
    problematico, o None. Funcion pura."""
    if valoracion(metrica, estado) != "a revisar":
        return None
    sentido = ABAJO if estado == "por debajo del rango" else ARRIBA
    for d in DIAGNOSTICOS:
        if d["metrica"] == metrica and d["sentido"] == sentido and golpe in d["golpes"]:
            return {k: d[k] for k in ("problema", "causas", "que_mirar", "ejercicios")}
    return None
