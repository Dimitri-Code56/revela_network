"""Dados: download e pré-processamento do MIT-Adobe FiveK, degradações sintéticas e um dataset sintético de teste.

Cada foto do FiveK vira um .npz pequeno com:
  lin     float16 (h, w, 3)  imagem linear da câmera, primárias sRGB, SEM balanço de branco (0..1 = preto..saturação)
  alvo_X  uint8   (h, w, 3)  revelação do especialista X (a..e) em sRGB
  wb      float32 (3,)       ganhos de balanço de branco "as shot" da câmera, no mesmo espaço de `lin`
  iso, exp float32           ISO e tempo de exposição (EXIF)
"""
import io
import json
import math
import os
import random
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cores import prophoto_to_srgb

LISTAS = "https://huggingface.co/datasets/yangkun2197/MIT-Adobe-FiveK/resolve/main/{}.json"
SPLITS = {"treino": "training", "validacao": "validation", "teste": "testing", "debug": "debugging"}
ESTILOS = "abcde"


# ------------------------------------------------------------------ download e pré-processamento
def _baixar(url, destino, tentativas=3):
    url = url.replace("http://", "https://")
    for t in range(tentativas):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(destino, "wb") as f:
                while True:
                    bloco = r.read(1 << 20)
                    if not bloco:
                        break
                    f.write(bloco)
            return
        except Exception:
            if t == tentativas - 1:
                raise


def baixar_lista(split):
    with urllib.request.urlopen(LISTAS.format(SPLITS[split]), timeout=60) as r:
        return json.loads(r.read())


def _redimensionar(img, h, w):
    t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
    return F.interpolate(t, size=(h, w), mode="bilinear", antialias=True, align_corners=False)[0].permute(1, 2, 0).numpy()


def _exif(caminho):
    import exifread
    with open(caminho, "rb") as f:
        tags = exifread.process_file(f, details=False)
    def num(chave, padrao):
        v = tags.get(chave)
        if v is None:
            return padrao
        x = v.values[0]
        return float(x.num) / float(x.den) if hasattr(x, "num") else float(x)
    return num("EXIF ISOSpeedRatings", 100.0), num("EXIF ExposureTime", 1 / 60)


def processar_foto(nome, info, pasta, lado=512, estilos=ESTILOS):
    """Baixa o DNG e os TIFFs dos especialistas, reduz e salva <pasta>/<nome>.npz. Devolve o caminho ou None."""
    import rawpy
    import tifffile
    saida = os.path.join(pasta, nome + ".npz")
    if os.path.exists(saida):
        return saida
    with tempfile.TemporaryDirectory() as tmp:
        dng = os.path.join(tmp, "foto.dng")
        _baixar(info["urls"]["dng"], dng)
        iso, exp = _exif(dng)
        kw = dict(half_size=True, output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1),
                  no_auto_bright=True, output_bps=16, use_auto_wb=False)
        with rawpy.imread(dng) as r:
            lin = r.postprocess(use_camera_wb=False, user_wb=[1, 1, 1, 1], **kw).astype(np.float32) / 65535
        with rawpy.imread(dng) as r:
            bal = r.postprocess(use_camera_wb=True, **kw).astype(np.float32) / 65535
        # ganhos efetivos de WB no espaço de `lin`, medidos em pixels bem expostos
        ok = (lin.max(-1) < 0.9) & (lin.min(-1) > 0.01)
        if ok.sum() < 1000:
            return None
        wb = np.median(bal[ok] / lin[ok], axis=0)
        wb = (wb / wb[1]).astype(np.float32)

        h0, w0 = lin.shape[:2]
        s = lado / max(h0, w0)
        h, w = max(8, round(h0 * s)), max(8, round(w0 * s))
        dados = {"lin": _redimensionar(lin, h, w).astype(np.float16), "wb": wb,
                 "iso": np.float32(iso), "exp": np.float32(exp)}
        for e in estilos:
            tif = os.path.join(tmp, f"{e}.tif")
            _baixar(info["urls"]["tiff16"][e], tif)
            alvo = tifffile.imread(tif)
            alvo = alvo[..., :3].astype(np.float32) / (65535 if alvo.dtype == np.uint16 else 255)
            if (alvo.shape[0] > alvo.shape[1]) != (h0 > w0):      # orientação diferente: descarta
                return None
            dados["alvo_" + e] = (np.clip(prophoto_to_srgb(_redimensionar(alvo, h, w)), 0, 1) * 255 + 0.5).astype(np.uint8)
            os.remove(tif)
        np.savez_compressed(saida, **dados)
    return saida


def preparar_fivek(pasta, split="treino", n=None, lado=512, estilos=ESTILOS, workers=8, semente=0):
    """Baixa e pré-processa `n` fotos de um split. Pode ser interrompido e retomado."""
    os.makedirs(pasta, exist_ok=True)
    lista = baixar_lista(split)
    nomes = sorted(lista)
    random.Random(semente).shuffle(nomes)
    nomes = nomes[:n] if n else nomes
    feitos, falhas = 0, 0
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **k: x
    with ThreadPoolExecutor(workers) as ex:
        futuros = [ex.submit(processar_foto, k, lista[k], pasta, lado, estilos) for k in nomes]
        for fut in tqdm(as_completed(futuros), total=len(futuros), desc=f"FiveK {split}"):
            try:
                feitos += fut.result() is not None
            except Exception:
                falhas += 1
    print(f"{split}: {feitos} fotos prontas, {len(nomes) - feitos} descartadas ({falhas} com erro de download/leitura)")


# ------------------------------------------------------------------ datasets
class FiveK(Dataset):
    """Lê os .npz. Em treino devolve recortes aleatórios de `recorte`×`recorte`; em avaliação, a foto inteira."""
    def __init__(self, pasta, recorte=384, treino=True):
        self.arquivos = sorted(os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".npz"))
        self.recorte, self.treino = recorte, treino
        if not self.arquivos:
            raise FileNotFoundError(f"nenhum .npz em {pasta}")

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        d = np.load(self.arquivos[i])
        lin = torch.from_numpy(d["lin"].astype(np.float32)).permute(2, 0, 1)
        alvos = {k[5:]: torch.from_numpy(d[k]).permute(2, 0, 1).float() / 255 for k in d.files if k.startswith("alvo_")}
        meta = torch.tensor([math.log2(float(d["iso"]) / 100), math.log2(max(float(d["exp"]), 1e-5)), 0, 0, 0, 0])
        wb = torch.from_numpy(d["wb"])
        if self.treino:
            lin, alvos = _recortar(lin, alvos, self.recorte)
        return {"lin": lin, "meta": meta, "wb": wb, "alvos": alvos}


def _recortar(lin, alvos, r):
    _, h, w = lin.shape
    if h < r or w < r:
        s = r / min(h, w)
        tam = (max(r, math.ceil(h * s)), max(r, math.ceil(w * s)))   # tamanho explícito: o arredondamento não pode deixar um lado menor que r
        lin = F.interpolate(lin[None], size=tam, mode="bilinear")[0]
        alvos = {k: F.interpolate(v[None], size=tam, mode="bilinear")[0] for k, v in alvos.items()}
        _, h, w = lin.shape
    y, x = random.randint(0, h - r), random.randint(0, w - r)
    flip = random.random() < 0.5
    corta = lambda t: (t[:, y:y + r, x:x + r].flip(-1) if flip else t[:, y:y + r, x:x + r])
    return corta(lin), {k: corta(v) for k, v in alvos.items()}


class Sintetico(Dataset):
    """Cenas sintéticas com 5 'especialistas' feitos de receitas conhecidas. Serve para testar o código sem baixar nada."""
    def __init__(self, n=64, tamanho=256, semente=0):
        self.n, self.t, self.semente = n, tamanho, semente

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.semente * 100003 + i)
        t = self.t
        yy, xx = torch.meshgrid(torch.linspace(0, 1, t), torch.linspace(0, 1, t), indexing="ij")
        refl = torch.rand(3, 1, 1, generator=g) * 0.3 + 0.1 + 0.2 * yy * torch.rand(3, 1, 1, generator=g)
        for _ in range(6):
            cx, cy, r = torch.rand(3, generator=g)
            cor = torch.rand(3, 1, 1, generator=g) * 0.8 + 0.05
            m = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (0.02 + 0.05 * r))
            refl = refl * (1 - m) + cor * m
        refl = refl * (1 + 0.05 * torch.randn(3, t, t, generator=g))
        luz = torch.tensor([1.0, 1.0, 1.0]) * torch.exp(torch.randn(3, generator=g) * 0.25)
        luz = luz / luz[1]
        lin = (refl * luz[:, None, None] * 0.6).clamp(0, 1)
        meta = torch.tensor([float(torch.randint(0, 5, (1,), generator=g)), -6.0, 0, 0, 0, 0])
        neutro = lin / luz[:, None, None]
        alvos = {e: _receita(neutro, e) for e in ESTILOS}
        return {"lin": lin, "meta": meta, "wb": (1 / luz).float(), "alvos": alvos}


def _receita(neutro, e):
    x = neutro.clamp(0, 1) ** (1 / 2.2)
    if e == "a":
        x = x ** 0.9
    elif e == "b":
        x = 0.5 + (x - 0.5) * 1.3
    elif e == "c":
        x = x * torch.tensor([1.05, 1.0, 0.92])[:, None, None]
    elif e == "d":
        l = x.mean(0, keepdim=True); x = l + (x - l) * 0.5
    else:
        x = 0.06 + 0.9 * x * torch.tensor([0.95, 1.0, 1.08])[:, None, None]
    return x.clamp(0, 1)


# ------------------------------------------------------------------ degradações sintéticas (etapa 1)
def degradar(lin, meta):
    """Sobre-expõe e adiciona ruído com parâmetros conhecidos. Devolve entrada e alvos das cabeças de cena."""
    B, dev = lin.shape[0], lin.device
    ganho = 2 ** (torch.rand(B, 1, 1, 1, device=dev) * 3)                   # 0 a 3 stops
    cena = lin * ganho
    log_a = torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(1e-4), math.log(1e-2))
    log_b = torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(1e-7), math.log(1e-4))
    var = log_a.exp() * cena.clamp(0, 1) + log_b.exp()
    entrada = (cena + torch.randn_like(cena) * var.sqrt()).clamp(0, 1)
    meta = meta.clone()
    meta[:, 0] = (log_a.flatten() - math.log(1e-4)) / math.log(2)          # "ISO" relativo ao ganho do sensor
    return {
        "entrada": entrada,
        "meta": meta,
        "cena": cena.clamp(max=16),
        "mascara": (cena.max(1, keepdim=True).values >= 1).float(),
        "log_sigma": 0.5 * var.log().mean(1, keepdim=True),
    }
