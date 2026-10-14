# Datos de referencia

- `manifest.csv`: clips de YouTube que baja `scripts/download_references.py`,
  con el jugador de cada uno. La columna `vista` es la del título; la vista
  real de cada golpe está en los ficheros de golpes.
- `golpes.csv`: impactos anotados a mano mirando los frames. `tolerancia` es la
  incertidumbre de la anotación en frames. Es la verdad contra la que se evalúa
  `segmentar_golpes.py` (`evaluar_segmentacion.py`).
- `golpes_auto.csv`: golpes cuyo impacto puso el segmentador y cuyo tipo se
  verificó a ojo. Se usan como referencias de métricas, nunca para evaluar al
  segmentador.

Vista "lateral": cámara a nivel de pista en el lateral. En esos clips el
jugador queda de frente en un tipo de golpe y de espaldas en el otro.

| Clip | Anotado |
|---|---|
| fh_djokovic_1000fps, fh_dimitrov_lat, fh_kyrgios, fh_roddick_lat, bh_djokovic_1000fps, sv_roddick, sv_serena, sv_zverev_lat, rt_sinner_lat, rt_alcaraz_lat | todo el clip, a mano |
| fh_compilacion_espalda | frames 0–1400, a mano |
| rt_djokovic_lat, rt_alcaraz2_lat, rt_lehecka_lat, rt_ymer_lat, rt_nava_lat | automático + tipo a ojo |
| sv_federer_lat, sv_alcaraz_tec, sv_osaka | automático + tipo y vista a ojo (2026-10-14) |
| fh_ruud, bh_dimitrov_lat, bh_zverev, rt_fils_lat | nada |
| rt_zheng_lat | excluido: cámara elevada por detrás |

Los reveses cortados de rt_alcaraz2_lat están etiquetados `reves_cortado` y no
entran en ninguna comparación. Todos los jugadores son diestros. La vista de cada golpe es la del momento
del impacto (Serena está de frente antes de sacar pero de perfil al golpear).
