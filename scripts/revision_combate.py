#!/usr/bin/env python3
"""Pagina de revision del analisis de combate: la silueta de cada clip sobre el
video, los golpes detectados y que se mide en cada uno. Se abre con doble clic.

    python scripts/revision_combate.py
    open data/revision_combate/index.html

Lee los clips de data/raw/combate/pro (profesionales) y de data/raw/combate
(propios o de prueba), con su esqueleto corregido en data/processed/combate/...
y escribe data/revision_combate/. data/ no esta en el repositorio: la pagina
lleva video de personas reales y se queda en local.

Por clip: el video con la silueta (esqueleto, silueta rellena o ambos, y la
opcion de fondo negro para ver solo el movimiento), una linea de tiempo
clicable con los golpes detectados, una ficha por golpe con sus metricas, la
calidad del clip y lo que dicen las reglas de correccion. La pestana "Estado"
dice que esta hecho, que esta verificado y donde se puede cambiar algo.

Para los profesionales se usa un perfil de referencia (180 cm, 75 kg,
avanzado): sirve para ver que detecta y que mide el sistema sobre gente que
golpea bien, no para juzgarlos.
"""

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analizar_combate as ac  # noqa: E402
import combate_perfil as cp  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "data", "revision_combate")
PERFIL_REFERENCIA = cp.Perfil(altura_cm=180, peso_kg=75, experiencia_meses=36)
# Vista comprobada manualmente; el resto sigue siendo una hipotesis lateral.
VISTAS_REVISADAS = {"5ZbVy-Kiayc": "frontal"}
LADO_VIDEO = 960
ESCALA = 10000
# nombre de landmark MediaPipe -> posicion en el array exportado
PUNTOS = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle", "left_foot_index", "right_foot_index"]


def recodificar(origen, destino):
    """H.264 sin audio, lado mayor LADO_VIDEO: lo que reproduce cualquier navegador."""
    import imageio_ffmpeg
    escala = f"scale='if(gt(iw,ih),min({LADO_VIDEO},iw),-2)':'if(gt(iw,ih),-2,min({LADO_VIDEO},ih))'"
    fd, temporal = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(destino))
    os.close(fd)
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", origen,
                        "-vf", escala, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "27",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporal], check=True)
        os.replace(temporal, destino)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)


def exportar_esqueleto(xy, vis, ancho, alto):
    """Lista (n, K*2) de enteros 0..ESCALA (-1 = sin dato) y visibilidades 0..100."""
    n = len(xy["nose"])
    puntos = np.full((n, len(PUNTOS), 2), -1, dtype=int)
    visib = np.zeros((n, len(PUNTOS)), dtype=int)
    for k, nombre in enumerate(PUNTOS):
        p = xy[nombre]
        ok = ~np.isnan(p).any(axis=1)
        puntos[ok, k, 0] = np.clip(p[ok, 0] / ancho * ESCALA, 0, ESCALA).astype(int)
        puntos[ok, k, 1] = np.clip(p[ok, 1] / alto * ESCALA, 0, ESCALA).astype(int)
        visib[:, k] = np.nan_to_num(np.asarray(vis[nombre], float) * 100, nan=0).astype(int)
    return puntos.reshape(n, -1).tolist(), visib.tolist()


def limpiar(v):
    if isinstance(v, dict):
        return {k: limpiar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [limpiar(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def datos_clip(nombre, video, procesado, fuente, perfil=PERFIL_REFERENCIA):
    """Analiza un clip y devuelve el dict que lee la pagina (o None si no hay esqueleto)."""
    import metricas_golpe
    from evaluar_esqueleto import cargar, detectar_saltos_persona, longitud_torso
    csv = os.path.join(procesado, f"{nombre}_corregido.csv")
    if not os.path.isfile(csv):
        return None
    fps, ancho, alto = metricas_golpe.info_video(video)
    n, xy, vis = cargar(csv, ancho, alto)
    vista = VISTAS_REVISADAS.get(nombre, "lateral")
    saltos = detectar_saltos_persona(xy, longitud_torso(xy))
    r = ac.analizar_xy(xy, vis, fps, perfil, vista, saltos)
    puntos, visib = exportar_esqueleto(xy, vis, ancho, alto)
    golpes = []
    from combate_golpes import detectar
    crudos = detectar(xy, fps, perfil.lado_delantero, saltos,
                      vista != "frontal", vis, detectar_patadas=vista != "frontal")
    for g, f in zip(crudos, r["golpes"]):
        golpes.append({**{k: g[k] for k in ("tipo", "lado", "inicio", "pico", "fin", "ref")},
                       "t_s": g["t_s"], "metricas": f})
    cortes = []
    ruta_cortes = os.path.join(procesado, f"{nombre}_cortes.csv")
    if os.path.isfile(ruta_cortes):
        import pandas as pd
        cortes = [int(x) for x in pd.read_csv(ruta_cortes)["frame"]]
    d1 = None
    ruta_d1 = os.path.join(procesado, f"{nombre}_d1.json")
    if os.path.isfile(ruta_d1):
        import combate_d1
        try:
            with open(ruta_d1, encoding="utf-8") as f:
                candidato = json.load(f)
            if candidato.get("video_sha256") == combate_d1.huella(video):
                d1 = candidato
            else:
                print(f"  {nombre}: revisión D1 ignorada: corresponde a otro vídeo")
        except (OSError, ValueError, AttributeError):
            print(f"  {nombre}: no se pudo leer la revisión D1")
    return limpiar({
        "nombre": nombre, "fuente": fuente, "fps": fps, "ancho": ancho, "alto": alto, "n": n,
        "puntos": puntos, "vis": visib, "cortes": cortes, "golpes": golpes,
        "sesion": r["sesion"], "calidad": r["calidad"], "reglas": [
            {k: x.get(k) for k in ("id", "problema", "estado", "valor", "unidad", "umbral_ok",
                                   "umbral_mal", "n", "motivo", "peor_t_s")} for x in r["reglas"]],
        "patadas": r["patadas"], "visible": ac.visibles(vis), "vista": vista,
        "notas": r["notas"], "d1": d1,
    })


def descubrir(raiz_videos, raiz_procesado, fuente):
    clips = []
    if not os.path.isdir(raiz_videos):
        return clips
    for f in sorted(os.listdir(raiz_videos)):
        if f.endswith(".mp4"):
            nombre = f[:-4]
            clips.append((nombre, os.path.join(raiz_videos, f), raiz_procesado, fuente))
    return clips


def escribir(clips, estado, salida=SALIDA, recodificar_video=True):
    os.makedirs(os.path.join(salida, "videos"), exist_ok=True)
    os.makedirs(os.path.join(salida, "datos"), exist_ok=True)
    indice = []
    for nombre, video, procesado, fuente in clips:
        d = datos_clip(nombre, video, procesado, fuente)
        if d is None:
            continue
        destino = os.path.join(salida, "videos", f"{nombre}.mp4")
        if recodificar_video and not os.path.isfile(destino):
            recodificar(video, destino)
        with open(os.path.join(salida, "datos", f"{nombre}.js"), "w", encoding="utf-8") as f:
            f.write(f"window.CLIPS=window.CLIPS||{{}};window.CLIPS[{json.dumps(nombre)}]="
                    + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + ";")
        s = d["sesion"]
        indice.append({"nombre": nombre, "fuente": fuente, "fps": d["fps"], "duracion_s": s["duracion_s"],
                       "puños": s["puños"], "patadas": s["patadas"], "fiables": s["fiables"],
                       "calidad": d["calidad"], "validos_pct": s["frames_validos_pct"]})
        print(f"  {nombre}: {s['puños']} puños, {s['patadas']} patadas, "
              f"{'OK' if not d['calidad'] else 'avisos: ' + str(len(d['calidad']))}")
    with open(os.path.join(salida, "indice.js"), "w", encoding="utf-8") as f:
        f.write("window.INDICE=" + json.dumps(limpiar(indice), ensure_ascii=False) + ";"
                "window.ESTADO=" + json.dumps(estado, ensure_ascii=False) + ";")
    with open(os.path.join(salida, "index.html"), "w", encoding="utf-8") as f:
        f.write(PLANTILLA)
    return os.path.join(salida, "index.html")


def estado_por_defecto():
    ruta = os.path.join(RAIZ, "docs", "estado_combate.json")
    if os.path.isfile(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {"hecho": [], "verificado": [], "abierto": [], "decisiones": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", default=SALIDA)
    parser.add_argument("--sin-video", action="store_true", help="no recodifica los videos")
    args = parser.parse_args()
    clips = descubrir(os.path.join(RAIZ, "data", "raw", "combate", "pro"),
                      os.path.join(RAIZ, "data", "processed", "combate", "pro"), "profesional")
    clips += descubrir(os.path.join(RAIZ, "data", "raw", "combate"),
                       os.path.join(RAIZ, "data", "processed", "combate"), "prueba")
    print(f"{len(clips)} clips")
    ruta = escribir(clips, estado_por_defecto(), args.salida, not args.sin_video)
    print(f"Pagina -> {ruta}")


PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Silueta de combate</title>
<style>

:root{--bg:#f1f0eb;--panel:#fff;--ink:#1b2527;--mut:#576668;--line:#d9dfdc;--acc:#aa422d;--punch:#176f91;--kick:#b84c2f;--ok:#286349;--warn:#815413;--bad:#9d372d;--sil:#f4f5ef}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}::selection{background:#cbe1d8;color:#182724}button:focus-visible,.tl:focus-visible{outline:3px solid #25806a;outline-offset:3px}button{touch-action:manipulation}
header{display:flex;gap:20px;align-items:center;padding:16px max(24px,calc((100vw - 1540px)/2));background:#182824;color:#f4f5ef;position:sticky;top:0;z-index:5;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}.brand-mark{display:grid;place-items:center;width:37px;height:37px;border:1px solid #a0bcb0;border-radius:50%;font-weight:800;font-size:15px}h1{font-size:17px;margin:0;line-height:1.15;letter-spacing:-.02em}header .mut{font-size:13px;color:#c0d0c8}nav{display:flex;gap:5px;margin-left:auto}
nav button,.btn{font:inherit;border:1px solid var(--line);background:transparent;color:var(--ink);padding:8px 12px;border-radius:7px;cursor:pointer}nav button{border-color:#536761;color:#f2f4f1;font-size:13px}nav button:hover{background:#30433b}nav button[aria-pressed=true]{background:#f4f5ef;color:#182824;border-color:#f4f5ef}.btn:hover{border-color:#576f65}.btn[aria-pressed=true]{background:#1e3930;color:#fff;border-color:#1e3930}
main{display:grid;grid-template-columns:250px minmax(0,1fr);gap:28px;padding:30px 24px 56px;max-width:1540px;margin:0 auto}.contenido{min-width:0}.lista{display:flex;flex-direction:column;gap:7px;max-height:calc(100vh - 118px);overflow:auto;position:sticky;top:94px;align-self:start}.lista h2{font-size:19px;margin:0 0 2px;letter-spacing:-.02em}.lista>p{color:var(--mut);font-size:13px;margin:0 0 10px}.item{border:1px solid transparent;border-bottom-color:var(--line);background:transparent;border-radius:8px;padding:12px;cursor:pointer;text-align:left;color:inherit;font:inherit}.item:hover{background:#e5eae4}.item[aria-current=true]{border-color:#a8bfb3;background:#e4ece6}.item b{display:block;font-size:14px;overflow-wrap:anywhere}.item span{font-size:13px;color:var(--mut)}
.chip{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);margin:7px 4px 0 0}.chip.ok{color:var(--ok);border-color:#a6c6b2}.chip.warn{color:var(--warn);border-color:#d8bf91}.chip.bad{color:var(--bad);border-color:#d6afa7}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:16px;min-width:0;overflow-x:auto}.card h2{font-size:16px;margin:0 0 14px;letter-spacing:-.01em}.intro{display:flex;justify-content:space-between;align-items:end;gap:20px;margin:0 0 22px}.intro h2{font-size:clamp(26px,3vw,38px);line-height:1.1;letter-spacing:-.03em;margin:0 0 7px}.intro p{margin:0;color:var(--mut);font-size:14px;max-width:68ch}.intro .estado-local{white-space:nowrap;font-size:13px;color:#385c4d;background:#dce9df;padding:7px 10px;border-radius:5px}
.reproductor{padding:0;overflow:hidden}.reproductor-head{display:flex;gap:12px;justify-content:space-between;align-items:center;padding:17px 20px}.reproductor-head h2{margin:0;overflow-wrap:anywhere}.reproductor-head span{font-size:13px;color:var(--mut);white-space:nowrap}.visor{position:relative;background:#111b18;overflow:hidden;line-height:0;width:100%;display:flex;justify-content:center}.visor video{width:auto;max-width:100%;max-height:65vh;height:auto;display:block}.visor canvas{position:absolute;inset:0;margin:auto;width:auto;height:100%;max-width:100%;pointer-events:none}.player-body{padding:16px 20px 20px}.ctrl{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px;align-items:center}.ctrl .btn{font-size:13px}.ctrl .sep{width:1px;height:20px;background:var(--line);margin:0 3px}
.timeline-head{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:var(--mut);margin-bottom:7px}.timeline-head strong{color:var(--ink)}.tl{position:relative;height:48px;border:1px solid var(--line);border-radius:7px;background:#f7f8f5;cursor:pointer;overflow:hidden}.tl .m{position:absolute;top:6px;height:14px;border-radius:2px;opacity:.9}.tl .m.p{background:var(--punch)}.tl .m.k{background:var(--kick);top:25px}.tl .c{position:absolute;top:0;bottom:0;width:1px;background:var(--bad);opacity:.5}.tl .cur{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);z-index:1}.tl small{display:none}.leyenda{font-size:13px;color:var(--mut);margin:8px 0 0}.leyenda span{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 5px 1px 12px}.leyenda span:first-child{margin-left:0}
.golpes{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}.g{border:1px solid var(--line);border-radius:8px;padding:12px;cursor:pointer;background:#fbfcfa;text-align:left;font:inherit;color:inherit}.g:hover{border-color:var(--acc)}.g .t{font-weight:650}.g.p .t{color:var(--punch)}.g.k .t{color:var(--kick)}.g .d{font-size:13px;color:var(--mut);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}.num{font-variant-numeric:tabular-nums;text-align:right}.est-bien{color:var(--ok)}.est-mejorar{color:var(--warn)}.est-prioritario{color:var(--bad)}.est-sin{color:var(--mut)}
ul{margin:4px 0 4px 18px;padding:0}li{margin:6px 0}.mut{color:var(--mut)}code{background:#e8ede9;padding:1px 4px;border-radius:4px;font-size:13px}.aviso{padding:10px 12px;background:#faf1df;border-radius:6px;margin:10px 0;font-size:13px;color:#65430c}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}.vacio{padding:40px;text-align:center;color:var(--mut)}.guia{margin-top:18px;padding:16px 12px;border-top:1px solid var(--line);font-size:13px;color:var(--mut)}.guia summary{cursor:pointer;color:var(--ink);font-weight:650}.guia p{margin:10px 0 0}.guia ol{padding-left:18px}.guia code{overflow-wrap:anywhere}
.selector-clips{display:none}
@media(max-width:900px){header{position:static;padding:14px 18px}main{grid-template-columns:minmax(0,1fr);padding:22px 16px 40px;gap:18px}.lista{position:static;max-height:none;min-width:0}.lista .item{display:none}.selector-clips{display:block;width:100%;min-width:0;padding:10px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--ink);font:inherit}.selector-clips:focus-visible{outline:3px solid #25806a;outline-offset:3px}.guia{margin-top:5px;padding-top:10px}.intro{align-items:start}.visor video{max-height:60vh}}
@media(max-width:540px){header{gap:12px}header .mut{display:none}nav{width:100%;margin:0}nav button{flex:1;min-width:0}.intro{display:block}.intro .estado-local{display:inline-block;margin-top:12px}.card{padding:16px}.reproductor{padding:0}.reproductor-head{align-items:start;flex-direction:column;gap:2px}.player-body{padding:14px}.ctrl .sep{display:none}.visor canvas{height:100%}.timeline-head{flex-wrap:wrap}table{font-size:13px}td,th{padding:8px 4px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body>
<header><div class="brand"><span class="brand-mark" aria-hidden="true">C</span><div><h1>Combate</h1><span class="mut">Análisis de movimiento</span></div></div><span class="mut" id="sub"></span>
<nav aria-label="Secciones"><button id="t-clips" aria-pressed="true">Revisión de vídeo</button><button id="t-estado" aria-pressed="false">Estado del proyecto</button></nav></header>
<main id="main"></main>
<script src="indice.js"></script>
<script>
const $=(s,r=document)=>r.querySelector(s), el=(t,a={},...c)=>{const e=document.createElement(t);for(const k in a){if(k==='class')e.className=a[k];else if(k.startsWith('on'))e[k]=a[k];else e.setAttribute(k,a[k])}c.flat().forEach(x=>e.append(x&&x.nodeType?x:document.createTextNode(x??'')));return e};
const PUNTOS=["nose","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_foot_index","right_foot_index"];
const I=Object.fromEntries(PUNTOS.map((n,i)=>[n,i]));
const HUESOS=[["left_shoulder","right_shoulder"],["left_hip","right_hip"],["left_shoulder","left_hip"],["right_shoulder","right_hip"],["left_shoulder","left_elbow"],["left_elbow","left_wrist"],["right_shoulder","right_elbow"],["right_elbow","right_wrist"],["left_hip","left_knee"],["left_knee","left_ankle"],["left_ankle","left_foot_index"],["right_hip","right_knee"],["right_knee","right_ankle"],["right_ankle","right_foot_index"]];
const MIEMBROS=[["left_shoulder","left_elbow",.30],["left_elbow","left_wrist",.24],["right_shoulder","right_elbow",.30],["right_elbow","right_wrist",.24],["left_hip","left_knee",.40],["left_knee","left_ankle",.30],["right_hip","right_knee",.40],["right_knee","right_ankle",.30],["left_ankle","left_foot_index",.14],["right_ankle","right_foot_index",.14]];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let actual=null, modo='ambos', fondo='video', clip=null, velocidad=1;
const fmt=(v,d=1)=>v==null?'-':Number(v).toFixed(d);
const ES={bien:'est-bien',"a mejorar":'est-mejorar',prioritario:'est-prioritario',"sin dato":'est-sin'};

function pt(f,n,W,H){const v=clip.puntos[f];if(!v)return null;const x=v[I[n]*2],y=v[I[n]*2+1];return x<0?null:[x/10000*W,y/10000*H]}
function dibujar(c,f){
  const W=c.width,H=c.height,x=c.getContext('2d');x.clearRect(0,0,W,H);
  if(fondo==='negro'){x.fillStyle='#0b0c0f';x.fillRect(0,0,W,H)}
  f=Math.max(0,Math.min(clip.n-1,f));
  const P=n=>pt(f,n,W,H),sh=[P('left_shoulder'),P('right_shoulder')],hp=[P('left_hip'),P('right_hip')];
  if(!sh[0]||!sh[1]||!hp[0]||!hp[1])return;
  const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2],ms=mid(sh[0],sh[1]),mh=mid(hp[0],hp[1]),torso=Math.hypot(ms[0]-mh[0],ms[1]-mh[1]);
  const g=clip.golpes.find(g=>f>=g.inicio&&f<=g.fin);
  const activo=g?(g.tipo==='puño'?[g.lado+'_wrist',g.lado+'_elbow',g.lado+'_shoulder']:[g.lado+'_ankle',g.lado+'_knee',g.lado+'_hip']):[];
  const col=g?(g.tipo==='puño'?css('--punch'):css('--kick')):css('--acc');
  if(modo!=='esqueleto'){
    x.lineCap='round';x.lineJoin='round';x.globalAlpha=fondo==='negro'?.95:.55;
    x.fillStyle=fondo==='negro'?css('--sil'):'#fff';x.strokeStyle=x.fillStyle;
    x.beginPath();x.moveTo(...sh[0]);x.lineTo(...sh[1]);x.lineTo(...hp[1]);x.lineTo(...hp[0]);x.closePath();x.fill();
    for(const [a,b,w] of MIEMBROS){const A=P(a),B=P(b);if(!A||!B)continue;x.lineWidth=w*torso*.55;x.beginPath();x.moveTo(...A);x.lineTo(...B);x.stroke()}
    const nz=P('nose');if(nz){x.beginPath();x.arc(nz[0],nz[1],torso*.17,0,7);x.fill()}
    x.globalAlpha=1;
  }
  if(modo!=='silueta'){
    x.lineWidth=Math.max(2,torso*.03);x.lineCap='round';
    for(const [a,b] of HUESOS){const A=P(a),B=P(b);if(!A||!B)continue;const act=activo.includes(a)&&activo.includes(b);
      x.strokeStyle=act?col:(a.startsWith('left')&&b.startsWith('left')?'#37b24d':a.startsWith('right')&&b.startsWith('right')?'#f03e3e':'#fab005');x.lineWidth=act?Math.max(4,torso*.06):Math.max(2,torso*.03);x.beginPath();x.moveTo(...A);x.lineTo(...B);x.stroke()}
    for(let i=0;i<PUNTOS.length;i++){const v=clip.puntos[f];if(v[i*2]<0)continue;const lo=clip.vis[f][i]<50;x.fillStyle=lo?'rgba(255,255,255,.35)':'#fff';x.beginPath();x.arc(v[i*2]/10000*W,v[i*2+1]/10000*H,lo?2.5:3.5,0,7);x.fill()}
  }
  if(g){x.font=`600 ${Math.max(14,W/48)}px system-ui`;x.fillStyle=col;const t=`${(g.metricas&&g.metricas.nombre)||g.tipo} ${g.lado==='left'?'izq.':'der.'}`;const tw=x.measureText(t).width;x.fillRect(10,10,tw+16,Math.max(14,W/48)+12);x.fillStyle='#fff';x.fillText(t,18,10+Math.max(14,W/48)+1)}
}
function vista(nombre){
  const cargar=()=>new Promise((r,rechazar)=>{if(window.CLIPS&&window.CLIPS[nombre])return r();const s=document.createElement('script');s.src=`datos/${nombre}.js`;s.onload=r;s.onerror=rechazar;document.head.append(s)});
  const main=$('#main');main.replaceChildren();
  const selector=el('select',{class:'selector-clips','aria-label':'Seleccionar vídeo',onchange:e=>{actual=e.target.value;vista(actual)}},INDICE.map(i=>el('option',{value:i.nombre,...(i.nombre===nombre?{selected:''}:{})},`${i.nombre} · ${i.fuente}`)));
  const lista=el('aside',{class:'lista','aria-label':'Biblioteca de vídeos'},el('h2',{},'Tus vídeos'),el('p',{},'Selecciona un clip procesado para revisarlo.'),selector,INDICE.map(i=>{
    const q=i.calidad.length?(i.fiables?'warn':'bad'):'ok';
    return el('button',{class:'item','aria-current':i.nombre===nombre,onclick:()=>{actual=i.nombre;vista(i.nombre)}},
      el('b',{},i.nombre),el('span',{},`${i.fuente} · ${fmt(i.duracion_s,0)} s · ${Math.round(i.fps)} fps`),
      el('div',{},el('span',{class:'chip '+q},q==='ok'?'Sin avisos de calidad':i.calidad.length+' aviso(s)')))}),
    el('details',{class:'guia'},el('summary',{},'Preparar un vídeo'),el('p',{},'Graba con la cámara estable y procura que se vea el cuerpo entero durante el ejercicio.'),el('p',{},'Esta página muestra vídeos ya procesados en este equipo. La incorporación y el análisis se realizan fuera del navegador; todavía no hay subida desde aquí.'),el('p',{},'Consulta las instrucciones de combate en el README del proyecto.')));
  const der=el('div',{class:'contenido'});main.append(lista,der);
  if(!nombre){der.append(el('div',{class:'vacio'},'Aún no hay vídeos procesados. Consulta cómo prepararlos en la guía.'));return}
  cargar().then(()=>{
    if(!der.isConnected)return;
    clip=CLIPS[nombre];const v=el('video',{src:`videos/${nombre}.mp4`,muted:'',playsinline:'',preload:'auto'}),c=el('canvas');
    const visor=el('div',{class:'visor'},v,c);
    const cur=el('div',{class:'cur'}),tl=el('div',{class:'tl',role:'slider',tabindex:'0','aria-label':'Posición del vídeo','aria-valuemin':'0','aria-valuemax':String(clip.n-1),'aria-valuenow':'0'},cur,clip.cortes.map(f=>el('div',{class:'c',style:`left:${f/clip.n*100}%`})),clip.golpes.map(g=>el('div',{class:'m '+(g.tipo==='puño'?'p':'k'),style:`left:${g.inicio/clip.n*100}%;width:${Math.max(.4,(g.fin-g.inicio)/clip.n*100)}%`,title:`${g.tipo} ${g.lado} ${g.t_s}s`})),el('small',{},'azul = puño · naranja = patada · rojo = corte'));
    const ir=f=>{v.currentTime=f/clip.fps};
    tl.onclick=e=>{const r=tl.getBoundingClientRect();ir(Math.max(0,Math.min(clip.n-1,Math.floor((e.clientX-r.left)/r.width*clip.n))))};tl.onkeydown=e=>{const f=Math.round(v.currentTime*clip.fps);if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();ir(Math.max(0,Math.min(clip.n-1,f+(e.key==='ArrowRight'?1:-1))))}};
    const refrescar=()=>{c.width=v.videoWidth||clip.ancho;c.height=v.videoHeight||clip.alto;const f=Math.round(v.currentTime*clip.fps);dibujar(c,f);cur.style.left=(f/clip.n*100)+'%';tl.setAttribute('aria-valuenow',String(Math.max(0,Math.min(clip.n-1,f))));tl.setAttribute('aria-valuetext',fmt(v.currentTime,1)+' segundos')};
    v.addEventListener('loadedmetadata',refrescar);v.addEventListener('seeked',refrescar);v.addEventListener('timeupdate',refrescar);
    if('requestVideoFrameCallback' in v){const cb=()=>{if(!v.isConnected)return;refrescar();v.requestVideoFrameCallback(cb)};v.requestVideoFrameCallback(cb)}
    const bt=(txt,fn,pressed)=>{const attrs={class:'btn',onclick:()=>fn(b)};if(pressed!==undefined)attrs['aria-pressed']=pressed;const b=el('button',attrs,txt);return b};
    const grupo=(items,get,set)=>{const bs=items.map(([k,t])=>bt(t,()=>{set(k);bs.forEach((b,i)=>b.setAttribute('aria-pressed',items[i][0]===k));refrescar()},get()===k));return bs};
    const ctrl=el('div',{class:'ctrl'},bt('Reproducir / pausar',()=>v.paused?v.play():v.pause()),bt('Fotograma anterior',()=>ir(Math.max(0,Math.round(v.currentTime*clip.fps)-1))),bt('Fotograma siguiente',()=>ir(Math.min(clip.n-1,Math.round(v.currentTime*clip.fps)+1))),
      el('span',{class:'sep'}),...grupo([[.25,'0,25×'],[.5,'0,5×'],[1,'1×']],()=>velocidad,k=>{velocidad=k;v.playbackRate=k}),el('span',{class:'sep'}),
      ...grupo([['ambos','Silueta + esqueleto'],['silueta','Silueta'],['esqueleto','Esqueleto']],()=>modo,k=>modo=k),el('span',{class:'sep'}),
      ...grupo([['video','Con vídeo'],['negro','Fondo negro']],()=>fondo,k=>{fondo=k;v.style.visibility=k==='negro'?'hidden':'visible'}));
    v.playbackRate=velocidad;v.style.visibility=fondo==='negro'?'hidden':'visible';
    const s=clip.sesion,vis=clip.visible,N={left_wrist:'muñeca izq.',right_wrist:'muñeca der.',left_ankle:'tobillo izq.',right_ankle:'tobillo der.'};
    const ficha=el('div',{class:'card'},el('h2',{},'Ficha'),el('table',{},
      ...[['Fuente',clip.fuente],['Resolución / fps',`${clip.ancho}×${clip.alto} · ${fmt(clip.fps,1)}`],['Duración',fmt(s.duracion_s,0)+' s'],['Persona detectada',fmt(s.frames_validos_pct,0)+' % de los fotogramas'],['Cortes de escena',clip.cortes.length],
        ...Object.keys(N).map(k=>['Se ve la '+N[k],fmt(vis[k],0)+' %']),['Guardia arriba',fmt(s.guardia_arriba_pct,0)+' % del tiempo'],['Ritmo',`${fmt(s.golpes_por_min,0)} golpes/min · ${s.combinaciones} combinaciones (máx. ${s.combinacion_mas_larga})`]].map(([a,b])=>el('tr',{},el('th',{},a),el('td',{},b)))),
      ...clip.calidad.map(m=>el('div',{class:'aviso'},'Aviso: '+m)),
      ...(clip.notas||[]).filter(n=>n.startsWith('Vista frontal')).map(n=>el('div',{class:'aviso'},n)));
    const reglas=el('div',{class:'card'},el('h2',{},'Qué dirían las reglas (perfil de referencia)'),el('table',{},el('tr',{},el('th',{},'Regla'),el('th',{class:'num'},'Valor'),el('th',{},'Estado')),
      ...clip.reglas.map(r=>el('tr',{},el('td',{},r.problema),el('td',{class:'num'},r.valor==null?'-':fmt(r.valor)+' '+(r.unidad||'').split(' ')[0]),el('td',{class:ES[r.estado]},r.estado+(r.motivo?' · '+r.motivo:''))))));
    const d1=clip.d1?el('div',{class:'card'},el('h2',{},'Revisión de encuadre · D1 Ultra'),
      el('p',{class:'mut'},'Probabilidades orientativas para los fotogramas muestreados; no modifican las métricas ni han sido validadas.'),
      el('table',{},el('tr',{},el('th',{},'Segundo'),el('th',{},'Cuerpo entero'),el('th',{},'Varias personas'),el('th',{},'De frente')),
        ...clip.d1.fotogramas.map(f=>el('tr',{},el('td',{},fmt(f.segundos,1)+' s'),...['cuerpo_entero','varias_personas','vista_frontal'].map(k=>el('td',{},fmt(f.probabilidades[k]*100,0)+' %')))))):null;
    const lg=el('div',{class:'card'},el('h2',{},`Golpes detectados (${clip.golpes.length})`),clip.golpes.length?el('div',{class:'golpes'},clip.golpes.map((g,i)=>{const m=g.metricas;
      const d=g.tipo==='puño'?`vel ${fmt(m.velocidad_ms)} m/s · ext ${fmt(m.extension_pct,0)} % · guardia ${fmt(m.guardia_otra_mano_cm,0)} cm · retorno ${fmt(m.retorno_guardia_ms,0)} ms`:`vel ${fmt(m.velocidad_ms)} m/s · ${m.zona||'-'} (${fmt(m.altura_cm,0)} cm) · retorno ${fmt(m.retorno_pierna_ms,0)} ms · tronco ${fmt(m.inclinacion_atras_deg,0)}°`;
      return el('button',{class:'g '+(g.tipo==='puño'?'p':'k'),onclick:()=>{v.pause();ir(g.pico)}},el('div',{class:'t'},`${i+1}. ${m.nombre||g.tipo} ${g.lado==='left'?'izq.':'der.'}${m.fiable?'':' (dato dudoso)'}`),el('div',{class:'d'},`${g.t_s} s`),el('div',{class:'d'},d))})):el('div',{class:'mut'},'Ningún golpe detectado.'));
    der.append(el('div',{class:'intro'},el('div',{},el('h2',{},'Cada movimiento cuenta'),el('p',{},'Revisa el vídeo, explora los golpes detectados y comprueba la calidad del análisis.')),el('span',{class:'estado-local'},'Revisión local')),el('div',{class:'card reproductor'},el('div',{class:'reproductor-head'},el('h2',{},nombre),el('span',{},`${fmt(s.duracion_s,0)} segundos · ${Math.round(clip.fps)} fps`)),visor,el('div',{class:'player-body'},ctrl,el('div',{class:'timeline-head'},el('strong',{},'Línea de tiempo'),el('span',{},'Selecciona un momento para revisarlo')),tl,el('p',{class:'leyenda'},el('span',{style:'background:var(--punch)'}),'Puño',el('span',{style:'background:var(--kick)'}),'Patada',el('span',{style:'background:var(--bad)'}),'Corte'))),el('div',{class:'cols'},ficha,reglas));
    if(d1)der.append(d1);der.append(lg);
    refrescar();
  }).catch(()=>{der.replaceChildren(el('div',{class:'vacio'},'No se pudieron cargar los datos del vídeo. Vuelve a generar la página de revisión.'))})}
function estado(){
  const E=window.ESTADO||{},main=$('#main');main.replaceChildren();
  const sec=(t,items)=>el('div',{class:'card'},el('h2',{},t),el('ul',{},(items||[]).map(i=>el('li',{},typeof i==='string'?i:i.titulo+(i.detalle?' — '+i.detalle:'')))));
  main.style.gridTemplateColumns='1fr';
  main.append(el('div',{class:'cols'},sec('Hecho',E.hecho),sec('Verificado (y cómo)',E.verificado)),el('div',{class:'cols'},sec('Abierto / no validado',E.abierto),sec('Dónde puedes decidir o cambiar',E.decisiones)))}
function tab(t){$('#t-clips').setAttribute('aria-pressed',t==='clips');$('#t-estado').setAttribute('aria-pressed',t==='estado');$('#main').style.gridTemplateColumns='';t==='clips'?vista(actual||(INDICE[0]&&INDICE[0].nombre)):estado()}
$('#t-clips').onclick=()=>tab('clips');$('#t-estado').onclick=()=>tab('estado');
$('#sub').textContent=`${INDICE.length} clips · ${INDICE.reduce((a,i)=>a+i.puños+i.patadas,0)} golpes detectados`;
tab('clips');
</script></body></html>
"""

if __name__ == "__main__":
    main()
