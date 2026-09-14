#!/usr/bin/env bash
#
# Descarga los modelos de MediaPipe Pose Landmarker a models/.
#
# models/ esta en .gitignore (son ~36 MB de binarios), asi que una clonacion
# limpia no puede ejecutar nada hasta que se corra este script.
#
# Es idempotente: si el modelo ya esta y tiene un tamano razonable, no lo
# vuelve a descargar. Para forzar la redescarga, borra el fichero.
#
#   ./scripts/download_models.sh

set -euo pipefail

BASE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
DIR_MODELOS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

# Un .task valido pesa megabytes; si baja menos que esto es que nos han
# devuelto una pagina de error en vez del binario.
TAM_MINIMO=1000000

mkdir -p "$DIR_MODELOS"

tam_fichero() {
    # stat tiene sintaxis distinta en macOS (BSD) y en Linux (GNU).
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0
}

descargar_modelo() {
    local variante="$1"
    local destino="$DIR_MODELOS/pose_landmarker_${variante}.task"
    local url="$BASE_URL/pose_landmarker_${variante}/float16/latest/pose_landmarker_${variante}.task"

    if [ -f "$destino" ] && [ "$(tam_fichero "$destino")" -ge "$TAM_MINIMO" ]; then
        echo "  [ok]   ${variante}: ya esta en models/ ($(tam_fichero "$destino") bytes)"
        return 0
    fi

    echo "  [...]  ${variante}: descargando..."
    local temporal="${destino}.parcial"
    # Descargamos a un temporal para no dejar un .task a medias si esto falla.
    if ! curl -sS -L --fail -o "$temporal" "$url"; then
        rm -f "$temporal"
        echo "ERROR: fallo la descarga de ${variante} desde:" >&2
        echo "       $url" >&2
        echo "       Comprueba la conexion, o busca la URL actual en la" >&2
        echo "       documentacion de MediaPipe Pose Landmarker." >&2
        return 1
    fi

    local tam
    tam=$(tam_fichero "$temporal")
    if [ "$tam" -lt "$TAM_MINIMO" ]; then
        rm -f "$temporal"
        echo "ERROR: ${variante} se descargo pero solo pesa ${tam} bytes." >&2
        echo "       Se esperaban varios MB: probablemente la URL devolvio un" >&2
        echo "       error HTML en vez del modelo." >&2
        return 1
    fi

    mv "$temporal" "$destino"
    echo "  [ok]   ${variante}: descargado (${tam} bytes)"
}

echo "Modelos en: $DIR_MODELOS"
descargar_modelo heavy   # el que usa el pipeline
descargar_modelo lite    # el que usan los tests de integracion
echo "Listo."
