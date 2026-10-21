"""
Emulador de câmera modular: cena (A⁻¹) compartilhada + renderizador de câmera (B) com LoRA.

Entrada : imagem linear da câmera (B, 3, H, W), já demosaicada, sem balanço de branco, normalizada pelo
          nível de branco do sensor (0 = preto, 1 = saturação), e metadados (B, 6):
          [log2 ISO/100, log2 tempo de exposição, 0, 0, 0, 0] (posições livres reservadas).
Saída   : imagem sRGB (B, 3, H, W) no estilo escolhido + fatores de cena interpretáveis.

Blocos
  1. Backbone        U-Net leve em baixa resolução (256×256), FiLM com metadados no gargalo
  2. Cabeças de cena iluminante (codebook K=16 + resíduo), exposição (máscara de corte),
                     material (8 classes), ruído (log σ)
  3. Altas luzes     U-Net pequena que só atua dentro da máscara de corte
  4. Renderizador    pipeline físico diferenciável (parâmetros por câmera)
                     + HDRNet residual (grade bilateral) com LoRA por câmera
                     + efeitos de sensor (ruído, vinheta, nitidez); JPEG real na inferência
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

LOW = 256          # resolução do caminho de inferência da cena
K_ILLUM = 16       # tamanho do codebook de iluminantes
D_CODE = 32        # dimensão de cada código
N_MAT = 8          # classes de material
META = 6           # metadados de captura


# ---------------------------------------------------------------- LoRA
class LoRAConv(nn.Module):
    """Conv (ou Linear como conv 1×1) com adaptadores LoRA nomeados por câmera."""
    def __init__(self, cin, cout, k=1, stride=1, rank=4):
        super().__init__()
        self.base = nn.Conv2d(cin, cout, k, stride, k // 2)
        self.rank, self.k, self.stride = rank, k, stride
        self.down, self.up = nn.ModuleDict(), nn.ModuleDict()
        self.active = {}                       # {câmera: peso}, permite misturar estilos

    def add_camera(self, name):
        self.down[name] = nn.Conv2d(self.base.in_channels, self.rank, self.k, self.stride, self.k // 2, bias=False)
        self.up[name] = nn.Conv2d(self.rank, self.base.out_channels, 1, bias=False)
        nn.init.zeros_(self.up[name].weight)   # começa sem alterar a base

    def forward(self, x):
        y = self.base(x)
        for name, w in self.active.items():
            y = y + w * self.up[name](self.down[name](x))
        return y


def eh_lora(nome):
    """True para pesos de adaptadores LoRA (renderer....down.<estilo>.weight / up.<estilo>.weight)."""
    return re.search(r"\.(down|up)\.[^.]+\.weight$", nome) is not None and nome.startswith("renderer")


def set_cameras(module, **weights):
    for m in module.modules():
        if isinstance(m, LoRAConv):
            m.active = dict(weights)


def add_camera(module, name):
    for m in module.modules():
        if isinstance(m, LoRAConv):
            m.add_camera(name)


# ---------------------------------------------------------------- blocos básicos
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, 1, 1), nn.Conv2d(cout, cout, 3, 1, 1)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(h)) + self.skip(x))


class FiLM(nn.Module):
    """Condicionamento: um vetor ajusta escala e deslocamento de cada canal."""
    def __init__(self, dcond, ch):
        super().__init__()
        self.lin = nn.Linear(dcond, 2 * ch)

    def forward(self, x, c):
        g, b = self.lin(c).chunk(2, -1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


# ---------------------------------------------------------------- 1. backbone
class Backbone(nn.Module):
    """U-Net 4 níveis [32, 64, 128, 256] em 256×256. Devolve features 64×128×128 e um vetor global 256."""
    def __init__(self, cin=4):                 # RGB linear + log-luminância
        super().__init__()
        ch = [32, 64, 128, 256]
        self.stem = ResBlock(cin, ch[0])
        self.down = nn.ModuleList([ResBlock(ch[i], ch[i + 1]) for i in range(3)])
        self.film = FiLM(META, ch[3])
        self.mid = ResBlock(ch[3], ch[3])
        self.up = nn.ModuleList([ResBlock(ch[3 - i] + ch[2 - i], ch[2 - i]) for i in range(2)])
        self.out = nn.Conv2d(ch[1], 64, 1)

    def forward(self, x, meta):
        s0 = self.stem(x)                                  # 32 × 256
        s1 = self.down[0](F.avg_pool2d(s0, 2))             # 64 × 128
        s2 = self.down[1](F.avg_pool2d(s1, 2))             # 128 × 64
        s3 = self.down[2](F.avg_pool2d(s2, 2))             # 256 × 32
        s3 = self.mid(self.film(s3, meta))
        g = s3.mean((2, 3))                                # vetor global
        u = self.up[0](torch.cat([F.interpolate(s3, scale_factor=2, mode="bilinear"), s2], 1))  # 128 × 64
        u = self.up[1](torch.cat([F.interpolate(u, scale_factor=2, mode="bilinear"), s1], 1))   # 64 × 128
        return self.out(u), g


# ---------------------------------------------------------------- 2. cabeças de cena
class IlluminantHead(nn.Module):
    """Escolhe um código do codebook (tipo de luz) + resíduo contínuo em log-crominância."""
    def __init__(self):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(K_ILLUM, D_CODE) * 0.1)
        self.code_chroma = nn.Parameter(torch.zeros(K_ILLUM, 2))   # (log R/G, log B/G) de cada tipo
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, K_ILLUM + 2))

    def forward(self, g, tau=1.0, hard=False):
        out = self.mlp(g)
        logits, resid = out[:, :K_ILLUM], 0.1 * torch.tanh(out[:, K_ILLUM:])
        if not self.training:                               # uso: escolha determinística do código
            w = F.one_hot(logits.argmax(-1), K_ILLUM).to(logits.dtype)
        elif hard:
            w = F.gumbel_softmax(logits, tau, hard=True)
        else:
            w = F.softmax(logits / tau, -1)
        emb = w @ self.codebook                             # (B, 32)
        chroma = w @ self.code_chroma + resid               # (B, 2)
        return {"logits": logits, "code": emb, "chroma": chroma}


class MapHead(nn.Module):
    def __init__(self, cout, extra=0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(64 + extra, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, cout, 1))

    def forward(self, f):
        return self.net(f)


class SceneHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.illum = IlluminantHead()
        self.exposure = MapHead(1)                          # P(estourado) por pixel
        self.material = MapHead(N_MAT)                      # pele, céu, vegetação, ...
        self.noise = MapHead(1, extra=META)                 # log σ do ruído

    def forward(self, f, g, meta):
        m = meta[..., None, None].expand(-1, -1, *f.shape[2:])
        return {
            **self.illum(g),
            "clip": torch.sigmoid(self.exposure(f)),
            "material": F.softmax(self.material(f), 1),
            "log_sigma": self.noise(torch.cat([f, m], 1)),
        }


# ---------------------------------------------------------------- 3. reconstrução de altas luzes
class HighlightRecon(nn.Module):
    """U-Net pequena [32, 64, 128]. Só altera pixels dentro da máscara: x' = x + m · (r − x)."""
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = ResBlock(4, 32), ResBlock(32, 64), ResBlock(64, 128)
        self.d2, self.d1 = ResBlock(128 + 64, 64), ResBlock(64 + 32, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, rgb, mask):
        x = torch.cat([rgb, mask], 1)
        a = self.e1(x); b = self.e2(F.avg_pool2d(a, 2)); c = self.e3(F.avg_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, size=b.shape[2:], mode="bilinear"), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, size=a.shape[2:], mode="bilinear"), a], 1))
        r = rgb + F.softplus(self.out(a))                   # só pode adicionar luz
        return rgb + mask * (r - rgb)


# ---------------------------------------------------------------- 4a. pipeline físico diferenciável
class PhysicalPipeline(nn.Module):
    """~40 parâmetros interpretáveis por câmera. O balanço de branco imita o ERRO da câmera antiga:
    uma MLP pequena mapeia o iluminante inferido para os ganhos que aquela câmera escolheria."""
    def __init__(self, knots=16):
        super().__init__()
        self.awb = nn.Sequential(nn.Linear(D_CODE + 2, 32), nn.SiLU(), nn.Linear(32, 2))  # ganhos log R, log B
        self.ccm = nn.Parameter(torch.eye(3))                                             # matriz de cor
        self.ev_bias = nn.Parameter(torch.zeros(1))                                       # viés do fotômetro
        self.center_weight = nn.Parameter(torch.tensor(1.0))                              # medição ponderada ao centro
        self.tone = nn.Parameter(torch.zeros(knots))                                      # curva monotônica
        self.sat = nn.Parameter(torch.zeros(1))
        self.clip_knee = nn.Parameter(torch.tensor(4.0))                                  # dureza do corte

    def tone_curve(self, x):
        inc = F.softplus(self.tone + 1.0)                    # incrementos positivos → monotônica
        ys = torch.cat([inc.new_zeros(1), torch.cumsum(inc, 0)]); ys = ys / ys[-1]
        n = len(inc); t = x.clamp(0, 1) * n; i = t.floor().clamp(max=n - 1).long(); f = t - i
        return ys[i] * (1 - f) + ys[i + 1] * f

    def forward(self, lin, illum):
        B, _, H, W = lin.shape
        gains = torch.exp(self.awb(torch.cat([illum["code"], illum["chroma"]], -1)))       # (B, 2)
        wb = torch.stack([gains[:, 0], torch.ones_like(gains[:, 0]), gains[:, 1]], 1)[..., None, None]
        x = lin * wb
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        wmap = torch.exp(-self.center_weight.abs() * (xx ** 2 + yy ** 2))
        lum = x.mean(1)
        meter = (lum * wmap).sum((1, 2)) / wmap.sum()
        ev = torch.log2(0.18 / meter.clamp(min=1e-4)) + self.ev_bias                      # exposição da câmera-alvo
        x = x * torch.pow(2.0, ev)[:, None, None, None]
        x = torch.einsum("ij,bjhw->bihw", self.ccm, x)
        k = F.softplus(self.clip_knee)
        x = x - F.softplus(k * (x - 1)) / k                                                # corte suave → duro
        x = x.clamp(min=0) ** (1 / 2.2)
        l = x.mean(1, keepdim=True); x = l + (x - l) * torch.exp(self.sat)
        return self.tone_curve(x)


# ---------------------------------------------------------------- 4b. HDRNet residual com LoRA
class ResidualHDRNet(nn.Module):
    """Caminho de baixa resolução prevê uma grade bilateral 16×16×8 de transformações afins 3×4.
    Um guia por pixel escolhe a fatia da grade. Todas as convs têm LoRA por câmera."""
    def __init__(self, cond=D_CODE + 2 + N_MAT + 2, grid_d=8):
        super().__init__()
        self.grid_d = grid_d
        c = [16, 32, 64, 64]
        self.low = nn.ModuleList([LoRAConv(3 + 1 + N_MAT + 1, c[0], 3, 2)] +
                                 [LoRAConv(c[i], c[i + 1], 3, 2) for i in range(3)])   # 256 → 16
        self.film = FiLM(cond, c[3])
        self.loc = LoRAConv(c[3], c[3], 3)
        self.glob = nn.Sequential(nn.Linear(c[3], c[3]), nn.SiLU())
        self.coef = LoRAConv(c[3], 12 * grid_d, 1)
        self.guide = nn.Sequential(LoRAConv(3, 16, 1), nn.SiLU(), LoRAConv(16, 1, 1))

    def forward(self, img, maps, cond):
        B, _, H, W = img.shape
        lo = F.interpolate(torch.cat([img, maps], 1), size=(LOW, LOW), mode="bilinear")
        h = lo
        for conv in self.low:
            h = F.silu(conv(h))
        h = self.film(h, cond)
        h = F.silu(self.loc(h) + self.glob(h.mean((2, 3)))[..., None, None])
        grid = self.coef(h).view(B, 12, self.grid_d, 16, 16)                     # (B, 12, D, 16, 16)
        z = torch.sigmoid(self.guide(img))                                      # (B, 1, H, W)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=img.device), torch.linspace(-1, 1, W, device=img.device), indexing="ij")
        pos = torch.stack([xx.expand(B, -1, -1), yy.expand(B, -1, -1), z[:, 0] * 2 - 1], -1)[:, None]
        A = F.grid_sample(grid, pos, align_corners=True)[:, :, 0].view(B, 3, 4, H, W)
        resid = (A[:, :, :3] * img[:, None]).sum(2) + A[:, :, 3]
        return img + resid


# ---------------------------------------------------------------- 4c. efeitos de sensor e lente
class SensorFX(nn.Module):
    """Parâmetros medidos por câmera. Ruído Poisson-Gaussiano: σ² = a·x + b, com a, b função do ISO."""
    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor([-9.0, 0.9]))   # log a = c0 + c1·log ISO
        self.log_b = nn.Parameter(torch.tensor([-14.0, 1.6]))
        self.chroma_ratio = nn.Parameter(torch.tensor(0.5))
        self.vignette = nn.Parameter(torch.tensor(0.15))
        self.sharpen = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, meta, add_noise=True):
        B, _, H, W = x.shape
        blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
        x = x + self.sharpen * (x - blur)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        x = x * (1 - self.vignette * (xx ** 2 + yy ** 2) / 2)
        if add_noise:
            liso = meta[:, 0][:, None, None, None]
            a = torch.exp(self.log_a[0] + self.log_a[1] * liso); b = torch.exp(self.log_b[0] + self.log_b[1] * liso)
            s = torch.sqrt(a * x.clamp(min=0) + b)
            n = torch.randn_like(x[:, :1]) * s.mean(1, keepdim=True)
            c = torch.randn_like(x) * s * self.chroma_ratio
            x = x + n + c - c.mean(1, keepdim=True)
        return x.clamp(0, 1)


class CameraRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.physical = PhysicalPipeline()
        self.residual = ResidualHDRNet()
        self.sensor = SensorFX()

    def forward(self, lin, scene, meta, add_noise=True):
        H, W = lin.shape[2:]
        up = lambda t: F.interpolate(t, size=(H, W), mode="bilinear")
        x = self.physical(lin, scene)
        maps = torch.cat([up(scene["clip"]), up(scene["material"]), up(scene["log_sigma"])], 1)
        cond = torch.cat([scene["code"], scene["chroma"], scene["material"].mean((2, 3)), scene["clip"].mean((2, 3)), scene["log_sigma"].mean((2, 3))], -1)
        x = self.residual(x, maps, cond)
        return self.sensor(x, meta, add_noise)


# ---------------------------------------------------------------- modelo completo
def log_lum(lin):
    return torch.log2(lin.mean(1, keepdim=True).clamp(min=1e-4)) / 8 + 1


class CameraEmulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.heads = SceneHeads()
        self.highlights = HighlightRecon()
        self.renderer = CameraRenderer()

    def infer_scene(self, rgb, meta):
        lo = F.interpolate(torch.cat([rgb, log_lum(rgb)], 1), size=(LOW, LOW), mode="bilinear", antialias=True)
        f, g = self.backbone(lo, meta)
        scene = self.heads(f, g, meta)
        mask = F.interpolate(scene["clip"], size=rgb.shape[2:], mode="bilinear")
        scene["linear"] = self.highlights(rgb, mask)
        return scene

    def forward(self, rgb, meta, add_noise=False):
        scene = self.infer_scene(rgb, meta)
        return self.renderer(scene["linear"], scene, meta, add_noise), scene


# ---------------------------------------------------------------- teste rápido
if __name__ == "__main__":
    torch.manual_seed(0)
    m = CameraEmulator()
    add_camera(m.renderer, "cybershot_w80")
    add_camera(m.renderer, "filme_portra")

    def count(mod):
        return sum(p.numel() for n, p in mod.named_parameters() if not re.search(r"\.(down|up)\.[a-z_0-9]+\.weight$", "." + n) or "base" in n)

    for name, mod in [("Backbone", m.backbone), ("Cabeças de cena", m.heads), ("Altas luzes", m.highlights),
                      ("Pipeline físico", m.renderer.physical), ("HDRNet residual", m.renderer.residual),
                      ("Efeitos de sensor", m.renderer.sensor)]:
        print(f"{name:20s} {count(mod):>10,d} parâmetros")
    lora_total = sum(p.numel() for n, p in m.renderer.named_parameters() if ("down.cybershot" in n or "up.cybershot" in n))
    print(f"{'LoRA por câmera':20s} {lora_total:>10,d} parâmetros")

    raw = torch.rand(1, 3, 512, 384); meta = torch.zeros(1, META)
    m.eval(); set_cameras(m.renderer, cybershot_w80=0.7, filme_portra=0.3)
    with torch.no_grad():
        out, scene = m(raw, meta)
    print("saída", tuple(out.shape), "| clip", tuple(scene["clip"].shape), "| material", tuple(scene["material"].shape),
          "| código de luz", tuple(scene["code"].shape))
