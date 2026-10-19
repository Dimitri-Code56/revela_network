"""Treino em 3 etapas.

  cena  backbone + cabeças + altas luzes, com alvos sintéticos (sobre-exposição e ruído conhecidos)
        e o WB "as shot" da câmera como rótulo do iluminante
  base  renderizador (pipeline físico + HDRNet) aprendendo um estilo de referência; a cena fica congelada
  lora  tudo congelado; uma LoRA por estilo

Uso:
  python -m renderizador.treino --dados dados/treino --validacao dados/validacao --etapa tudo --saida pesos
  python -m renderizador.treino --sintetico --etapa tudo --iters 30      # teste rápido sem dados
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cores import delta_e76, psnr, srgb_to_lab_t
from .dados import FiveK, Sintetico, degradar
from .modelo import CameraEmulator, LoRAConv, add_camera, eh_lora, set_cameras


def lotes(loader):
    while True:
        for b in loader:
            yield b


def para(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else {kk: vv.to(dev) for kk, vv in v.items()}) for k, v in b.items()}


def perda_imagem(pred, alvo):
    return F.l1_loss(pred, alvo) + 0.01 * F.l1_loss(srgb_to_lab_t(pred), srgb_to_lab_t(alvo))


def log(msg, arq=None):
    print(msg, flush=True)
    if arq:
        with open(arq, "a") as f:
            f.write(msg + "\n")


# ------------------------------------------------------------------ etapa 1: cena
def etapa_cena(m, it_lotes, args, dev, registro):
    m.train()
    params = list(m.backbone.parameters()) + list(m.heads.parameters()) + list(m.highlights.parameters())
    opt = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        b = para(next(it_lotes), dev)
        d = degradar(b["lin"], b["meta"])
        s = m.infer_scene(d["entrada"], d["meta"])
        lo = s["clip"].shape[2:]
        l_ilum = F.l1_loss(s["chroma"], torch.log(b["wb"][:, [0, 2]].clamp(min=1e-3)))
        uso = F.softmax(s["logits"], -1).mean(0)
        l_div = (uso * uso.clamp(min=1e-8).log()).sum()                                  # usa vários códigos
        l_mask = F.binary_cross_entropy(s["clip"].clamp(1e-5, 1 - 1e-5), F.adaptive_max_pool2d(d["mascara"], lo))
        l_ruido = F.l1_loss(s["log_sigma"], F.adaptive_avg_pool2d(d["log_sigma"], lo))
        lrec = torch.log1p(s["linear"].clamp(min=0) * 8); lcena = torch.log1p(d["cena"] * 8)
        l_luz = (d["mascara"] * (lrec - lcena).abs()).sum() / d["mascara"].sum().clamp(min=1) + 0.1 * F.l1_loss(lrec, lcena)
        perda = l_ilum + 0.05 * l_div + l_mask + 0.5 * l_ruido + l_luz
        opt.zero_grad(); perda.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if it % args.log == 0 or it == args.iters:
            log(f"[cena] {it}/{args.iters} perda {perda.item():.4f} | ilum {l_ilum.item():.3f} máscara {l_mask.item():.3f} "
                f"ruído {l_ruido.item():.3f} altas-luzes {l_luz.item():.3f} | {time.time() - t0:.0f}s", registro)


def congelar_cena(m):
    for mod in (m.backbone, m.heads, m.highlights):
        for p in mod.parameters():
            p.requires_grad_(False)


# ------------------------------------------------------------------ etapa 2: renderizador base
def etapa_base(m, it_lotes, args, dev, registro):
    congelar_cena(m)
    for p in m.heads.material.parameters():          # material sem rótulo: aprende junto com o renderizador
        p.requires_grad_(True)
    set_cameras(m.renderer)
    params = [p for p in list(m.renderer.parameters()) + list(m.heads.material.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters)
    m.train(); t0 = time.time()
    for it in range(1, args.iters + 1):
        b = para(next(it_lotes), dev)
        ganho = 2 ** (torch.rand(b["lin"].shape[0], 1, 1, 1, device=dev) * 2 - 1)   # robustez a ±1 stop
        out, _ = m((b["lin"] * ganho).clamp(0, 1), b["meta"])
        perda = perda_imagem(out, b["alvos"][args.estilo_base])
        opt.zero_grad(); perda.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if it % args.log == 0 or it == args.iters:
            log(f"[base:{args.estilo_base}] {it}/{args.iters} perda {perda.item():.4f} | {time.time() - t0:.0f}s", registro)


# ------------------------------------------------------------------ etapa 3: LoRAs
def parametros_lora(m, nome):
    return [p for mod in m.renderer.modules() if isinstance(mod, LoRAConv)
            for p in list(mod.down[nome].parameters()) + list(mod.up[nome].parameters())]


def etapa_lora(m, it_lotes, args, dev, registro, nome):
    for p in m.parameters():
        p.requires_grad_(False)
    add_camera(m.renderer, nome)
    m.to(dev)
    params = parametros_lora(m, nome)
    for p in params:
        p.requires_grad_(True)
    set_cameras(m.renderer, **{nome: 1.0})
    opt = torch.optim.AdamW(params, args.lr_lora)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr_lora, total_steps=args.iters_lora)
    m.train(); t0 = time.time()
    for it in range(1, args.iters_lora + 1):
        b = para(next(it_lotes), dev)
        out, _ = m(b["lin"], b["meta"])
        perda = perda_imagem(out, b["alvos"][nome])
        opt.zero_grad(); perda.backward(); opt.step(); sched.step()
        if it % args.log == 0 or it == args.iters_lora:
            log(f"[lora:{nome}] {it}/{args.iters_lora} perda {perda.item():.4f} | {time.time() - t0:.0f}s", registro)
    torch.save({k: v.cpu() for k, v in m.renderer.state_dict().items() if f".{nome}." in k},
               os.path.join(args.saida, f"lora_{nome}.pt"))


# ------------------------------------------------------------------ avaliação
@torch.no_grad()
def avaliar(m, loader, estilos, base, dev, registro, max_fotos=100):
    m.eval()
    res = {}
    for nome in ["sem_lora"] + estilos:
        set_cameras(m.renderer, **({} if nome == "sem_lora" else {nome: 1.0}))
        alvos = ([base] + estilos) if nome == "sem_lora" else [nome]
        for alvo in alvos:
            ps, de = [], []
            for i, b in enumerate(loader):
                if i >= max_fotos:
                    break
                b = para(b, dev)
                out, _ = m(b["lin"], b["meta"])
                ps.append(psnr(out, b["alvos"][alvo]).item()); de.append(delta_e76(out, b["alvos"][alvo]).item())
            res[f"{nome}->{alvo}"] = {"psnr": sum(ps) / len(ps), "delta_e": sum(de) / len(de)}
    for k, v in res.items():
        log(f"[avaliação] {k:16s} PSNR {v['psnr']:.2f} dB | ΔE {v['delta_e']:.2f}", registro)
    set_cameras(m.renderer)
    return res


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados"); ap.add_argument("--validacao")
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--etapa", default="tudo", choices=["cena", "base", "lora", "tudo"])
    ap.add_argument("--estilo-base", default="c"); ap.add_argument("--estilos", default="a,b,d,e")
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--iters-lora", type=int)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--recorte", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr-lora", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--log", type=int, default=100)
    ap.add_argument("--saida", default="pesos"); ap.add_argument("--retomar")
    args = ap.parse_args(argv)
    args.iters_lora = args.iters_lora or max(1, args.iters // 4)
    estilos = [e for e in args.estilos.split(",") if e]
    os.makedirs(args.saida, exist_ok=True)
    registro = os.path.join(args.saida, "registro.txt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    if args.sintetico:
        treino, val = Sintetico(256, args.recorte), Sintetico(8, args.recorte, semente=1)
    else:
        treino, val = FiveK(args.dados, args.recorte, True), FiveK(args.validacao or args.dados, treino=False)
    it_lotes = lotes(DataLoader(treino, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                                pin_memory=dev == "cuda", persistent_workers=args.workers > 0))
    val_loader = DataLoader(val, 1, num_workers=args.workers)
    log(f"dispositivo {dev} | {len(treino)} fotos de treino, {len(val)} de validação", registro)

    m = CameraEmulator().to(dev)
    if args.retomar:
        m.load_state_dict(torch.load(args.retomar, map_location=dev), strict=False)
        log(f"pesos carregados de {args.retomar}", registro)

    if args.etapa in ("cena", "tudo"):
        etapa_cena(m, it_lotes, args, dev, registro)
        torch.save(m.state_dict(), os.path.join(args.saida, "cena.pt"))
    if args.etapa in ("base", "tudo"):
        etapa_base(m, it_lotes, args, dev, registro)
        torch.save({k: v for k, v in m.state_dict().items() if not eh_lora(k)},
                   os.path.join(args.saida, "base.pt"))
    if args.etapa in ("lora", "tudo"):
        for nome in estilos:
            etapa_lora(m, it_lotes, args, dev, registro, nome)
    if args.etapa in ("base", "lora", "tudo"):
        res = avaliar(m, val_loader, estilos if args.etapa != "base" else [], args.estilo_base, dev, registro)
        with open(os.path.join(args.saida, "metricas.json"), "w") as f:
            json.dump(res, f, indent=2)
    return m


if __name__ == "__main__":
    main()
