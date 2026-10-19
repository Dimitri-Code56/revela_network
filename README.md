# Renderizador de cena

Um modelo que separa **o que a cena é** de **como ela é desenhada**.

A primeira metade infere a cena a partir da imagem linear da câmera: qual luz iluminava, onde a foto estourou, que tipo de material aparece em cada região e quanto ruído existe. A segunda metade renderiza essa cena num estilo. Cada estilo (um fotógrafo, um filme, uma câmera antiga) é uma **LoRA de cerca de 8,5 mil parâmetros**, e estilos podem ser misturados.

O projeto começou como uma emulação do visual das Cyber-shot CCD e foi generalizado. A câmera antiga passa a ser só mais uma LoRA, treinada quando houver dados dela.

## Arquitetura

```
imagem linear (3×H×W) + metadados (ISO, exposição)
│
├─ INFERÊNCIA DA CENA (compartilhada entre estilos)
│   Backbone U-Net 32→64→128→256 em 256×256, FiLM com metadados ........ 3,2 M
│   Cabeças ............................................................. 93 mil
│     iluminante  codebook de 16 tipos de luz + resíduo contínuo
│     exposição   probabilidade de estouro por pixel (1×128×128)
│     material    8 classes por pixel, aprendidas sem rótulo
│     ruído       log σ por pixel
│   Altas luzes  U-Net pequena que só age dentro da máscara de estouro ..... 0,5 M
│
└─ RENDERIZADOR (específico do estilo)
    Pipeline físico diferenciável ........................................ 1,2 mil
      balanço de branco a partir do iluminante, fotômetro ponderado ao centro,
      matriz de cor 3×3, corte de altas luzes, curva de tom com 16 nós, saturação
    HDRNet residual: grade bilateral 16×16×8 de afins 3×4 ................ 115 mil
      todas as convs com LoRA por estilo (rank 4) ........................ 8,5 mil por estilo
    Efeitos de sensor: ruído σ² = a·x + b, vinheta, nitidez .............. 7
```

## Treino em 3 etapas

| Etapa | O que treina | Dados e alvos |
|---|---|---|
| `cena` | backbone, cabeças, altas luzes | FiveK com sobre-exposição e ruído sintéticos (alvos exatos); WB "as shot" da câmera como rótulo do iluminante |
| `base` | renderizador + cabeça de material | revelação do especialista C do FiveK |
| `lora` | só a LoRA de cada estilo | especialistas A, B, D e E |

O [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) tem 5.000 fotos RAW, cada uma revelada por 5 fotógrafos. Isso dá 5 estilos com pares perfeitos. As listas de treino, validação e teste com os links de cada arquivo vêm do [espelho no Hugging Face](https://huggingface.co/datasets/yangkun2197/MIT-Adobe-FiveK). O dataset é liberado para pesquisa; confira a licença antes de qualquer uso comercial.

## Como rodar

### No Colab (recomendado)
Abra `notebooks/treino_colab.ipynb` no Colab com GPU. Ele baixa e pré-processa os dados para o seu Drive, treina as 3 etapas e aplica o resultado numa foto sua.

### Localmente
```bash
pip install -r requirements.txt

# teste de ponta a ponta sem baixar nada (CPU, poucos minutos)
python scripts/teste_sintetico.py

# dados
python -c "from renderizador.dados import preparar_fivek as p; p('dados/validacao','validacao',n=60); p('dados/treino','treino',n=500)"

# treino
python -m renderizador.treino --dados dados/treino --validacao dados/validacao --etapa tudo --iters 15000 --saida pesos

# usar: uma mistura de estilos, ou uma grade com todos
python -m renderizador.usar foto.dng --pesos pesos --estilos a=0.7,e=0.3 --saida foto_estilo.jpg
python -m renderizador.usar foto.dng --pesos pesos --grade grade.jpg
```

## Estrutura

```
renderizador/
  modelo.py    arquitetura completa e LoRA (add_camera, set_cameras para misturar estilos)
  cores.py     ProPhoto → sRGB, Lab, ΔE, PSNR
  dados.py     download e pré-processamento do FiveK, degradações sintéticas, dataset sintético
  treino.py    as 3 etapas, checkpoints e avaliação
  usar.py      aplicar em fotos (DNG ou JPG) e gerar grades de comparação
notebooks/treino_colab.ipynb
scripts/teste_sintetico.py
prototipos/    as versões anteriores: app web do estilo CCD, receita CCD em Python, demo de LoRA
```

## Métricas

`treino.py` grava `metricas.json` com PSNR e ΔE (CIE76) na validação. A comparação que importa é `sem_lora->X` contra `X->X`: quanto a LoRA aproximou o estilo X além do que a base já fazia.

## Limitações conhecidas

- A cabeça de material não tem rótulo; as 8 classes são o que o renderizador achar útil, não necessariamente "pele" ou "céu".
- O rótulo do iluminante é o balanço de branco da própria câmera, que também erra.
- A entrada em JPEG é uma aproximação: o processamento do celular já foi aplicado.
- A textura por região (nitidez, redução de ruído e grão variando por material) ainda é global; os mapas de parâmetros e a CNN de textura são o próximo passo.
