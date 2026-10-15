import math, json, unicodedata
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

class KhmerTransformer(nn.Module):
    def __init__(self, chars):
        super().__init__()
        def block(i,o,p):
            return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),
                nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.MaxPool2d(p))
        self.cnn=nn.Sequential(block(1,64,(2,2)),block(64,128,(2,2)),
            block(128,256,(2,1)),block(256,512,(2,1)),nn.Conv2d(512,512,(4,1)),
            nn.BatchNorm2d(512),nn.ReLU(True))
        self.proj=nn.Linear(512,256)
        layer=nn.TransformerEncoderLayer(256,8,1024,dropout=0.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        # Each encoder layer must start with independent parameters.
        for layer in self.encoder.layers:
            for p in layer.parameters():
                if p.ndim > 1: nn.init.xavier_uniform_(p)
        self.norm=nn.LayerNorm(256)
        self.fc=nn.Linear(256,len(chars)+1)
        pos=torch.arange(128,dtype=torch.float32)[:,None]
        div=torch.exp(torch.arange(0,256,2)*(-math.log(10000.)/256))
        pe=torch.zeros(128,256); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('position',pe[None])
    def forward(self,x):
        x=self.proj(self.cnn(x).squeeze(2).permute(0,2,1))
        return self.fc(self.norm(self.encoder(x+self.position[:,:x.shape[1]])))




def clean(text):
    return ' '.join(unicodedata.normalize('NFC', str(text)).split())

class Recognizer:
    def __init__(self, checkpoint):
        torch.set_num_threads(2)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.chars = list(state['chars'])
        self.model = KhmerTransformer(self.chars).eval()
        self.model.load_state_dict(state['model'], strict=True)
        self.step = state.get('step')
        self.cer = state.get('cer')

    def predict(self, path):
        image = Image.open(path).convert('L')
        width = max(8, min(512, round(image.width * 64 / image.height)))
        canvas = Image.new('L', (512, 64), 255)
        canvas.paste(image.resize((width, 64), Image.Resampling.BILINEAR), (0, 0))
        x = torch.from_numpy((np.asarray(canvas, dtype=np.float32) / 255 - .5) / .5)[None, None]
        with torch.inference_mode():
            tokens = self.model(x).argmax(-1)[0].tolist()
        result, previous = [], -1
        for token in tokens:
            if token and token != previous:
                result.append(self.chars[token - 1])
            previous = token
        return clean(''.join(result))


class Recurrent(nn.Module):
 def __init__(self,chars,kind):
  super().__init__(); self.cnn=KhmerTransformer(chars).cnn
  self.rnn=(nn.GRU if kind=='bigru' else nn.LSTM)(512,256,2,batch_first=True,bidirectional=True,dropout=.1)
  self.fc=nn.Linear(512,len(chars)+1)
 def forward(self,x):
  x=self.cnn(x).squeeze(2).permute(0,2,1);x,_=self.rnn(x);return self.fc(x)

class MultiRecognizer(Recognizer):
    def __init__(self, checkpoint):
        torch.set_num_threads(2)
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.chars=c['chars']
        kind=c['architecture']
        self.model=(KhmerTransformer(self.chars) if kind=='transformer' else Recurrent(self.chars,kind)).eval()
        self.model.load_state_dict(c['model'],strict=True)

import unicodedata
COENG = '្'
INHERENT = {'឴', '឵'}
ZERO_WIDTH = {'\u200b', '\u200c', '\u200d', '\ufeff', '\xad'}

def _is_consonant(ch):
    return 'ក' <= ch <= 'អ'

def _is_dep_vowel(ch):
    return 'ា' <= ch <= 'ៅ'

def _is_shifter(ch):
    return ch in ('៉', '៊')

def _is_sign(ch):
    return ch in ('ំ', 'ះ', 'ៈ') or '់' <= ch <= '៑' or ch == '៓'

def _split_syllables(text):
    clusters, cur = ([], '')
    for ch in text:
        starts_new = _is_consonant(ch) or 'ឣ' <= ch <= 'ឳ'
        if starts_new and cur.endswith(COENG):
            starts_new = False
        if starts_new and cur:
            clusters.append(cur)
            cur = ch
        elif starts_new:
            cur = ch
        else:
            if not cur:
                clusters.append(ch)
                continue
            cur += ch
    if cur:
        clusters.append(cur)
    return clusters

def _canonical_cluster(cluster):
    if not cluster:
        return cluster
    base, i = (cluster[0], 1)
    coengs, shifters, vowels, signs = ([], [], [], [])
    while i < len(cluster):
        ch = cluster[i]
        if ch == COENG and i + 1 < len(cluster):
            coengs.append(COENG + cluster[i + 1])
            i += 2
            continue
        if _is_shifter(ch):
            shifters.append(ch)
        elif _is_dep_vowel(ch):
            vowels.append(ch)
        elif _is_sign(ch):
            signs.append(ch)
        else:
            signs.append(ch)
        i += 1
    coengs.sort()
    signs.sort()
    return base + ''.join(coengs) + ''.join(shifters) + ''.join(vowels) + ''.join(signs)

def normalize_khmer(text):
    if not text:
        return ''
    text = unicodedata.normalize('NFC', text)
    text = ''.join((c for c in text if c not in ZERO_WIDTH and c not in INHERENT))
    text = ''.join((_canonical_cluster(c) for c in _split_syllables(text)))
    return ' '.join(text.split())

class HandwritingRecognizer(Recognizer):
    def __init__(self,checkpoint):
        torch.set_num_threads(2)
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.chars=list(c['chars']);self.model=Recurrent(self.chars,'bilstm').eval()
        self.model.load_state_dict(c['model'],strict=True)
    def predict(self,path):
        im=Image.open(path).convert('L')
        width=max(8,min(512,int(im.width*64/max(im.height,1))))
        canvas=Image.new('L',(512,64),255)
        canvas.paste(im.resize((width,64),Image.Resampling.BILINEAR),(0,0))
        x=torch.from_numpy((np.asarray(canvas,dtype=np.float32)/255-.5)/.5)[None,None]
        with torch.inference_mode():tokens=self.model(x).argmax(-1)[0].tolist()
        out=[];previous=-1
        for token in tokens:
            if token and token!=previous:out.append(self.chars[token-1])
            previous=token
        return normalize_khmer(''.join(out))
