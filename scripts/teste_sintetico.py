"""Teste rápido de ponta a ponta, sem baixar nada (roda em CPU em poucos minutos).

  python scripts/teste_sintetico.py
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from renderizador.cores import prophoto_to_srgb  # noqa: E402
from renderizador.dados import FiveK  # noqa: E402
from renderizador import treino, usar  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    # 1. conversão de cor: branco ProPhoto tem que virar branco sRGB
    branco = prophoto_to_srgb(np.ones((1, 1, 3)))
    assert np.allclose(branco, 1, atol=2e-3), branco
    print("cores ok")

    # 2. leitura de um .npz no formato do pré-processamento
    pasta = os.path.join(tmp, "npz"); os.makedirs(pasta)
    np.savez_compressed(os.path.join(pasta, "x.npz"), lin=np.random.rand(300, 400, 3).astype(np.float16),
                        wb=np.array([2.0, 1.0, 1.5], np.float32), iso=np.float32(200), exp=np.float32(0.01),
                        **{f"alvo_{e}": (np.random.rand(300, 400, 3) * 255).astype(np.uint8) for e in "abcde"})
    item = FiveK(pasta, recorte=128)[0]
    assert item["lin"].shape == (3, 128, 128) and set(item["alvos"]) == set("abcde")
    print("dataset ok")

    # 3. treino das 3 etapas
    pesos = os.path.join(tmp, "pesos")
    treino.main(["--sintetico", "--etapa", "tudo", "--iters", "8", "--batch", "2", "--recorte", "96",
                 "--workers", "0", "--log", "4", "--estilos", "a,e", "--saida", pesos])

    # 4. uso: carregar pesos e renderizar uma foto com mistura de estilos
    foto = os.path.join(tmp, "foto.jpg")
    Image.fromarray((np.random.rand(120, 160, 3) * 255).astype(np.uint8)).save(foto)
    usar.main([foto, "--pesos", pesos, "--estilos", "a=0.5,e=0.5", "--saida", os.path.join(tmp, "out.jpg"),
               "--grade", os.path.join(tmp, "grade.jpg")])
    print("tudo ok")
