"""Aplicar o modelo treinado numa foto, com qualquer mistura de estilos.

  python -m renderizador.usar foto.dng --pesos pesos --estilos a=0.7,e=0.3 --saida foto_estilo.jpg
  python -m renderizador.usar foto.jpg --pesos pesos --grade grade.jpg      # todos os estilos lado a lado

DNG é o caminho certo (entrada linear de verdade). JPEG funciona como aproximação: a curva sRGB é desfeita,
mas o balanço de branco e o tom já aplicados pelo celular continuam lá.
"""
import argparse
import glob
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .modelo import CameraEmulator, add_camera, set_cameras


def carregar(pesos, dev="cpu"):
    m = CameraEmulator()
    estilos = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(pesos, "lora_*.pt")))
    for e in estilos:
        add_camera(m.renderer, e)
    estado = torch.load(os.path.join(pesos, "base.pt"), map_location=dev)
    for e in estilos:
        estado.update({"renderer." + k: v for k, v in torch.load(os.path.join(pesos, f"lora_{e}.pt"), map_location=dev).items()})
    m.load_state_dict(estado, strict=False)
    return m.to(dev).eval(), estilos


def ler_foto(caminho, lado=1024):
    if caminho.lower().endswith(".dng"):
        import rawpy
        with rawpy.imread(caminho) as r:
            lin = r.postprocess(half_size=True, output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1), no_auto_bright=True,
                                output_bps=16, use_auto_wb=False, use_camera_wb=False, user_wb=[1, 1, 1, 1]).astype(np.float32) / 65535
    else:
        x = np.asarray(Image.open(caminho).convert("RGB"), dtype=np.float32) / 255
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * 0.6
    t = torch.from_numpy(lin).permute(2, 0, 1)[None]
    s = min(1, lado / max(t.shape[2:]))
    if s < 1:
        t = torch.nn.functional.interpolate(t, scale_factor=s, mode="bilinear", antialias=True)
    return t


@torch.no_grad()
def renderizar(m, lin, **mistura):
    set_cameras(m.renderer, **mistura)
    out, _ = m(lin, torch.zeros(lin.shape[0], 6, device=lin.device))
    return out


def para_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def grade(m, estilos, lin, caminho):
    paineis = [("sem LoRA", {})] + [(e, {e: 1.0}) for e in estilos]
    if len(estilos) >= 2:
        paineis.append((f"50% {estilos[0]} + 50% {estilos[-1]}", {estilos[0]: .5, estilos[-1]: .5}))
    imgs = [(nome, para_pil(renderizar(m, lin, **mix))) for nome, mix in paineis]
    w, h = imgs[0][1].size; cols = min(3, len(imgs)); linhas = math.ceil(len(imgs) / cols)
    tela = Image.new("RGB", (cols * w, linhas * (h + 28)), (20, 20, 22))
    for i, (nome, im) in enumerate(imgs):
        x, y = (i % cols) * w, (i // cols) * (h + 28)
        tela.paste(im, (x, y + 28)); ImageDraw.Draw(tela).text((x + 6, y + 8), nome, fill=(235, 235, 230))
    tela.save(caminho, quality=92)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("foto"); ap.add_argument("--pesos", default="pesos")
    ap.add_argument("--estilos", default=""); ap.add_argument("--saida", default="saida.jpg")
    ap.add_argument("--grade"); ap.add_argument("--lado", type=int, default=1024)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, estilos = carregar(a.pesos, dev)
    lin = ler_foto(a.foto, a.lado).to(dev)
    if a.grade:
        grade(m, estilos, lin, a.grade); print("grade salva em", a.grade)
    mistura = {k: float(v) for k, v in (p.split("=") for p in a.estilos.split(",") if p)}
    para_pil(renderizar(m, lin, **mistura)).save(a.saida, quality=92)
    print("foto salva em", a.saida, "| estilos:", mistura or "nenhum (só a base)")


if __name__ == "__main__":
    main()
