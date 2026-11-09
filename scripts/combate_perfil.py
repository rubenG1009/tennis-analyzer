"""Perfil de la persona que se analiza (boxeo / K-1).

Lo introduce una persona: el video no dice cuanto pesas ni cuanto mides. Se usa
para tres cosas, y solo para esas:

1. Pasar las medidas del video (en "torsos") a unidades reales: cm y m/s. Con
   una sola camara es una aproximacion en el plano de la imagen.
2. Elegir cuanto se exige: un principiante, un intermedio y un avanzado no
   tienen los mismos umbrales (ver combate_correcciones.py).
3. Personalizar el texto: lo que hay que corregir, en tus centimetros, y las
   limitaciones que declares (lesiones, flexibilidad) para no recomendarte
   lo que no debes.

El peso NO se usa para estimar fuerza ni potencia absoluta: con video 2D no
hay forma honesta de hacerlo. Solo da el "momento del segmento" (masa del
brazo o la pierna x velocidad), util para comparar tus propios golpes entre si
y tu evolucion, no para compararte con otros.

Proporciones: Winter, "Biomechanics and Motor Control of Human Movement", tabla
antropometrica estandar (fraccion de la altura total y de la masa corporal).
Son medias poblacionales: una persona concreta se aparta unos puntos.
"""

import json
from dataclasses import asdict, dataclass, field

# Fraccion de la altura: hombro a cadera (0,818 - 0,530), hombro a muneca
# (brazo 0,186 + antebrazo 0,146) y cadera a tobillo (muslo 0,245 + pierna 0,246).
FRAC_TORSO = 0.818 - 0.530
FRAC_BRAZO = 0.186 + 0.146
FRAC_PIERNA = 0.245 + 0.246
# Fraccion de la masa corporal del segmento entero: brazo (0,028 + 0,016 + 0,006)
# y pierna (0,100 + 0,0465 + 0,0145).
MASA_BRAZO = 0.028 + 0.016 + 0.006
MASA_PIERNA = 0.100 + 0.0465 + 0.0145

POSTURAS = ("ortodoxa", "zurda")
DISCIPLINAS = ("boxeo", "k1")
FLEXIBILIDADES = ("baja", "media", "alta")
NIVELES = ("principiante", "intermedio", "avanzado")

# Convencion, no estandar: meses de practica regular.
MESES_INTERMEDIO = 6
MESES_AVANZADO = 24


@dataclass(frozen=True)
class Perfil:
    altura_cm: float
    peso_kg: float
    experiencia_meses: int = 0
    postura: str = "ortodoxa"        # ortodoxa: pierna y mano izquierdas delante
    disciplina: str = "k1"
    flexibilidad: str = "media"      # autoevaluada
    edad: int = None
    nombre: str = ""
    limitaciones: list = field(default_factory=list)  # texto libre: "rodilla derecha"...

    def __post_init__(self):
        if not 130 <= self.altura_cm <= 230:
            raise ValueError(f"altura_cm fuera de rango (130-230): {self.altura_cm}")
        if not 35 <= self.peso_kg <= 180:
            raise ValueError(f"peso_kg fuera de rango (35-180): {self.peso_kg}")
        if self.experiencia_meses < 0:
            raise ValueError("experiencia_meses no puede ser negativa")
        if self.edad is not None and not 10 <= self.edad <= 90:
            raise ValueError(f"edad fuera de rango (10-90): {self.edad}")
        for valor, validos, nombre in ((self.postura, POSTURAS, "postura"),
                                       (self.disciplina, DISCIPLINAS, "disciplina"),
                                       (self.flexibilidad, FLEXIBILIDADES, "flexibilidad")):
            if valor not in validos:
                raise ValueError(f"{nombre} debe ser una de {validos}, no {valor!r}")

    @property
    def nivel(self):
        if self.experiencia_meses < MESES_INTERMEDIO:
            return "principiante"
        if self.experiencia_meses < MESES_AVANZADO:
            return "intermedio"
        return "avanzado"

    @property
    def lado_delantero(self):
        """Lado del cuerpo (nombre de landmark) que va adelante en guardia."""
        return "left" if self.postura == "ortodoxa" else "right"

    @property
    def cm_por_torso(self):
        return FRAC_TORSO * self.altura_cm

    @property
    def brazo_cm(self):
        return FRAC_BRAZO * self.altura_cm

    @property
    def pierna_cm(self):
        return FRAC_PIERNA * self.altura_cm

    @property
    def imc(self):
        return self.peso_kg / (self.altura_cm / 100) ** 2

    @property
    def masa_brazo_kg(self):
        return MASA_BRAZO * self.peso_kg

    @property
    def masa_pierna_kg(self):
        return MASA_PIERNA * self.peso_kg

    def a_dict(self):
        return asdict(self)


def desde_dict(d):
    """Perfil desde un dict (por ejemplo, un JSON). Ignora claves desconocidas
    con un error claro en vez de en silencio."""
    permitidas = set(Perfil.__dataclass_fields__)
    sobran = set(d) - permitidas
    if sobran:
        raise ValueError(f"claves desconocidas en el perfil: {sorted(sobran)}")
    return Perfil(**d)


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return desde_dict(json.load(f))
