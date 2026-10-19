import numpy as np, torch, torch.nn as nn, math
from PIL import Image, ImageOps, ImageDraw, ImageFont
from estilo_ccd import look as ccd_look
torch.manual_seed(0); rng=np.random.default_rng(0)
SRC=['/root/.claude/uploads/5265526c-0a43-5ebc-8a80-05787d75df0c/e1414ab1-image.jpg','/root/.claude/uploads/5265526c-0a43-5ebc-8a80-05787d75df0c/e591230a-image.jpg']
def load(p,side):
    im=ImageOps.exif_transpose(Image.open(p)).convert('RGB'); s=side/max(im.size)
    return np.asarray(im.resize((round(im.width*s),round(im.height*s)),Image.LANCZOS),dtype=np.float32)/255
imgs=[load(p,384) for p in SRC]

def film_look(x):  # "filme quente": pretos levantados, altas suaves, verde contido
    l=(x@[0.2126,0.7152,0.0722])[...,None]
    y=x+(1-l)**2*np.array([-0.01,0.01,0.035])+l**2*np.array([0.035,0.01,-0.04])
    y=0.045+0.92*(1-np.exp(-2.2*y))/(1-math.exp(-2.2))
    l2=(y@[0.2126,0.7152,0.0722])[...,None]; y=l2+(y-l2)*0.85
    return np.clip(y,0,1).astype(np.float32)
STYLES={'ccd':lambda x:ccd_look(x).astype(np.float32),'filme':film_look}

def feats(x):
    l=x@[0.2126,0.7152,0.0722]
    return np.concatenate([x.reshape(-1,3).mean(0),x.reshape(-1,3).std(0),np.percentile(l,[5,25,50,75,95])]).astype(np.float32)

class LoRALinear(nn.Module):
    def __init__(s,i,o,r=4):
        super().__init__(); s.base=nn.Linear(i,o); s.r=r; s.A=nn.ParameterDict(); s.B=nn.ParameterDict(); s.mix={}
    def add(s,name):
        s.A[name]=nn.Parameter(torch.randn(s.r,s.base.in_features)*0.01); s.B[name]=nn.Parameter(torch.zeros(s.base.out_features,s.r))
    def forward(s,x):
        y=s.base(x)
        for n,w in s.mix.items():
            if w: y=y+w*(x@s.A[n].T@s.B[n].T)
        return y
class Net(nn.Module):  # rede base: pixel + contexto global da foto -> cor
    def __init__(s):
        super().__init__(); s.l=nn.ModuleList([LoRALinear(14,64),LoRALinear(64,64),LoRALinear(64,64),LoRALinear(64,3)])
    def forward(s,rgb,f):
        h=torch.cat([rgb,f.expand(rgb.shape[0],-1)],1)
        for i,l in enumerate(s.l):
            h=l(h); h=torch.nn.functional.silu(h) if i<3 else h
        return rgb+h
    def set_mix(s,**w):
        for l in s.l: l.mix=dict(w)

def perturb(x,strong):
    g=rng.uniform(*(0.8,1.25) if strong else (0.92,1.08),3); gm=rng.uniform(*(0.7,1.4) if strong else (0.88,1.12)); sat=rng.uniform(*(0.6,1.4) if strong else (0.9,1.1))
    y=np.clip(x*g,0,1)**gm; l=(y@[0.2126,0.7152,0.0722])[...,None]; return np.clip(l+(y-l)*sat,0,1).astype(np.float32)

def batch(target_fn,strong,n=4096):
    x=imgs[rng.integers(len(imgs))]; xi=perturb(x,strong); t=target_fn(xi,x)
    idx=rng.integers(0,xi.shape[0]*xi.shape[1],n)
    return torch.tensor(xi.reshape(-1,3)[idx]),torch.tensor(feats(xi)),torch.tensor(t.reshape(-1,3)[idx])

net=Net(); base_params=sum(p.numel() for l in net.l for p in l.base.parameters())
# 1) pré-treino da base: aprender a "revelar neutro" (desfazer erros de cor/tom)
opt=torch.optim.Adam([p for l in net.l for p in l.base.parameters()],2e-3)
for it in range(2500):
    x,f,t=batch(lambda xi,x:x,True); loss=(net(x,f)-t).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
print(f'base: {base_params} parâmetros, erro final {loss.item():.4f}')
for p in net.parameters(): p.requires_grad_(False)
# 2) uma LoRA por estilo, base congelada
for name,fn in STYLES.items():
    for l in net.l: l.add(name)
    ps=[p for l in net.l for p in (l.A[name],l.B[name])]
    for p in ps: p.requires_grad_(True)
    net.set_mix(**{name:1.0}); opt=torch.optim.Adam(ps,3e-3)
    for it in range(2000):
        x,f,t=batch(lambda xi,x:fn(xi),False); loss=(net(x,f)-t).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
    for p in ps: p.requires_grad_(False)
    print(f'LoRA "{name}": {sum(p.numel() for p in ps)} parâmetros, erro vs receita {loss.item()*255:.1f}/255 por canal')
torch.save(net.state_dict(),'lora_demo.pt')

def render(x,**mix):
    net.set_mix(**mix)
    with torch.no_grad(): y=net(torch.tensor(x.reshape(-1,3)),torch.tensor(feats(x))).numpy()
    return np.clip(y.reshape(x.shape),0,1)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26)
for k,p in enumerate(SRC):
    x=load(p,900); panels=[('Original',x),('Só a base',render(x)),('LoRA CCD',render(x,ccd=1.0)),('LoRA Filme',render(x,filme=1.0)),('50% CCD + 50% Filme',render(x,ccd=.5,filme=.5)),('Receita CCD (alvo)',ccd_look(x))]
    H,W,_=x.shape; cols=3; pad=10; grid=Image.new('RGB',(cols*W+(cols+1)*pad,2*(H+44)+3*pad),(24,24,26))
    for i,(t,img) in enumerate(panels):
        cx=pad+(i%cols)*(W+pad); cy=pad+(i//cols)*(H+44+pad)
        grid.paste(Image.fromarray((img*255+.5).astype(np.uint8)),(cx,cy+44)); ImageDraw.Draw(grid).text((cx+4,cy+8),t,font=font,fill=(235,235,230))
    grid.save(f'lora_demo_{k+1}.jpg',quality=90); print('salvo',f'lora_demo_{k+1}.jpg',grid.size)
