import sys, numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
def look(x):
    R,G,B=[np.power(x[...,i],2.2) for i in range(3)]
    R=R*0.95*1.08; G=G*0.99*1.08; B=B*1.07*1.08
    e=1e-4; rg=np.maximum(0,R-G); gr=np.maximum(0,G-R); bb=np.maximum(0,B-np.maximum(R,G))
    G=G*(1-0.18*rg/(R+e)); B=B*(1-0.12*rg/(R+e))
    B=B+0.06*gr; R=R*(1-0.10*gr/(G+e))
    R=R*(1-0.15*bb/(B+e)); G=G*(1-0.06*bb/(B+e))
    lum=0.2126*R+0.7152*G+0.0722*B
    out=[np.maximum(0,lum+(c-lum)*1.15) for c in (R,G,B)]
    def curve(v):
        v=np.clip(np.power(np.minimum(v,1.2),1/2.2),0,1)
        return np.where(v<0.5,0.5*np.power(2*v,1.18),1-0.5*np.power(2*(1-v),1.12))
    return np.clip(np.stack([curve(c) for c in out],-1),0,1)
def fringe(x,k):
    H,W,_=x.shape; cy,cx=(H-1)/2,(W-1)/2; yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    def samp(c,s):
        X=np.clip(cx+(xx-cx)*s,0,W-1); Y=np.clip(cy+(yy-cy)*s,0,H-1)
        x0=np.minimum(X.astype(int),W-2); y0=np.minimum(Y.astype(int),H-2); fx=X-x0; fy=Y-y0; ch=x[...,c]
        return (ch[y0,x0]*(1-fx)+ch[y0,x0+1]*fx)*(1-fy)+(ch[y0+1,x0]*(1-fx)+ch[y0+1,x0+1]*fx)*fy
    o=x.copy(); o[...,0]=samp(0,1/(1+k)); o[...,2]=samp(2,1/(1-k)); return o
def blur(x):
    w=[0.06,0.24,0.40,0.24,0.06]; p=np.pad(x,((2,2),(2,2),(0,0)),mode='edge'); H,W,_=x.shape
    t=sum(w[k]*p[2:2+H,k:k+W] for k in range(5)); p=np.pad(t,((2,2),(0,0),(0,0)),mode='edge')
    return sum(w[k]*p[k:k+H] for k in range(5))
def run(src,dst,res=2592,sharp=.55,noise=.35,vig=.30,ca=.30,jpeg=40,stamp=False,smear=False,date=None):
    im=ImageOps.exif_transpose(Image.open(src)).convert('RGB'); s=min(1,res/max(im.size))
    if s<1: im=im.resize((round(im.width*s),round(im.height*s)),Image.LANCZOS)
    x=np.asarray(im,dtype=np.float32)/255; H,W,_=x.shape
    x=look(x)
    if ca: x=fringe(x,ca*0.003)
    if sharp: x=x+(x-blur(x))*sharp*1.6
    yy,xx=np.mgrid[0:H,0:W]; d=np.hypot(xx-W/2,yy-H/2)/np.hypot(W/2,H/2); x=x*(1-vig*0.38*d*d)[...,None]
    rng=np.random.default_rng(7); l=np.clip(x@[0.299,0.587,0.114],0,1); sg=noise*(0.012+0.05*(1-l)**2)
    ln=rng.standard_normal((H,W))*sg; cn=rng.standard_normal((H,W))*sg*0.5
    x[...,0]+=ln+cn; x[...,1]+=ln; x[...,2]+=ln-cn
    if smear:
        frac=((x@[0.299,0.587,0.114])>0.995).sum(0)/H; st=np.minimum(0.3,frac*2.5); st=st+0.5*np.r_[0,st[:-1]]+0.5*np.r_[st[1:],0]
        x+=st[None,:,None]*np.array([0.95,0.82,1.0])
    out=Image.fromarray((np.clip(x,0,1)*255+0.5).astype(np.uint8))
    if stamp:
        fs=round(H*0.05); txt=date
        try: f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',fs)
        except: f=ImageFont.load_default()
        g=Image.new('RGBA',out.size,(0,0,0,0)); dr=ImageDraw.Draw(g); bw=dr.textlength(txt,font=f)
        pos=(W-H*0.045-bw,H-H*0.035-fs*1.1); dr.text(pos,txt,font=f,fill=(255,110,0,180))
        g=g.filter(ImageFilter.GaussianBlur(fs*0.12)); ImageDraw.Draw(g).text(pos,txt,font=f,fill=(255,141,30,255))
        out=Image.alpha_composite(out.convert('RGBA'),g).convert('RGB')
    out.save(dst,quality=round((0.95-jpeg*0.0045)*100))
    print(dst,out.size)
