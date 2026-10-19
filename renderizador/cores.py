"""Conversões de cor usadas nos dados e nas perdas."""
import numpy as np
import torch

# ProPhoto (ROMM, D50) -> XYZ D50
PROPHOTO_TO_XYZ_D50 = np.array([[0.7976749, 0.1351917, 0.0313534],
                                [0.2880402, 0.7118741, 0.0000857],
                                [0.0000000, 0.0000000, 0.8252100]])
# Adaptação cromática de Bradford, D50 -> D65
BRADFORD_D50_TO_D65 = np.array([[0.9555766, -0.0230393, 0.0631636],
                                [-0.0282895, 1.0099416, 0.0210077],
                                [0.0122982, -0.0204830, 1.3299098]])
# XYZ D65 -> sRGB linear
XYZ_D65_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]])
SRGB_TO_XYZ_D65 = np.linalg.inv(XYZ_D65_TO_SRGB)
PROPHOTO_TO_SRGB_LIN = XYZ_D65_TO_SRGB @ BRADFORD_D50_TO_D65 @ PROPHOTO_TO_XYZ_D50
D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def prophoto_decode(v):
    """Curva ROMM (gama 1.8 com trecho linear) -> linear."""
    v = np.clip(v, 0, 1)
    return np.where(v < 16 / 512, v / 16, v ** 1.8)


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def prophoto_to_srgb(img):
    """TIFF ProPhoto 16 bits normalizado [0,1] -> sRGB codificado [0,1]."""
    lin = prophoto_decode(img) @ PROPHOTO_TO_SRGB_LIN.T
    return srgb_encode(lin).astype(np.float32)


# ------------------------------------------------------------ versões em torch (perdas e métricas)
def _t(m, ref):
    return torch.as_tensor(m, dtype=ref.dtype, device=ref.device)


def srgb_to_linear_t(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb_to_lab_t(x):
    """(B,3,H,W) sRGB codificado -> Lab (B,3,H,W)."""
    lin = srgb_to_linear_t(x)
    xyz = torch.einsum("ij,bjhw->bihw", _t(SRGB_TO_XYZ_D65, x), lin) / _t(D65_WHITE, x)[None, :, None, None]
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-8) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


def delta_e76(pred, target):
    """Diferença de cor perceptual média (CIE76) entre duas imagens sRGB."""
    return (srgb_to_lab_t(pred) - srgb_to_lab_t(target)).norm(dim=1).mean()


def psnr(pred, target):
    mse = ((pred.clamp(0, 1) - target) ** 2).mean()
    return 10 * torch.log10(1 / mse.clamp(min=1e-10))
