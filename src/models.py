"""Two models, trained from scratch. No pretrained weights anywhere.

CRNN  -- the baseline. Its job is to prove the data pipeline is correct,
         cheaply (~2.5h). If CRNN cannot learn, the bug is in the labels.
AttnOCR -- the final model. Replaces CTC with an autoregressive attention
         decoder, because CTC assumes a monotonic left-to-right alignment
         between input columns and output characters. Khmer violates that:
         subscripts stack vertically under a base consonant and some vowels
         render to the LEFT of the consonant they follow in the encoding.
         Attention can attend anywhere and handles this; CTC cannot.
"""
import math

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Shared CNN. Input (B,1,64,W) -> (B, W/4, 512).

    Height collapses 64 -> 1; width divides by 4 only, so a 512px-wide crop
    yields 128 timesteps -- enough for CTC on our longest labels (~40 chars).
    """

    def __init__(self, out_dim=512):
        super().__init__()

        def blk(i, o, pool):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(pool),
            )

        self.net = nn.Sequential(
            blk(1, 64, (2, 2)),        # 64 -> 32, W/2
            blk(64, 128, (2, 2)),      # 32 -> 16, W/4
            blk(128, 256, (2, 1)),     # 16 -> 8,  W/4
            blk(256, out_dim, (2, 1)), # 8  -> 4,  W/4
            nn.Conv2d(out_dim, out_dim, (4, 1)),   # 4 -> 1
            nn.BatchNorm2d(out_dim), nn.ReLU(True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        f = self.net(x)              # (B,C,1,W')
        return f.squeeze(2).permute(0, 2, 1)   # (B,W',C)


class CRNN(nn.Module):
    def __init__(self, n_classes, hidden=256):
        super().__init__()
        self.backbone = Backbone()
        self.rnn = nn.LSTM(self.backbone.out_dim, hidden, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        f, _ = self.rnn(self.backbone(x))
        return self.head(f)          # (B,T,n_classes) -- CTC logits


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AttnOCR(nn.Module):
    """CNN encoder -> transformer encoder -> autoregressive transformer decoder.

    Token layout: 0 = pad, 1 = BOS, 2 = EOS, 3.. = characters.
    """

    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, n_chars, d=384, nhead=6, enc_layers=4, dec_layers=4):
        super().__init__()
        self.n_tokens = n_chars + 3
        self.backbone = Backbone(out_dim=d)
        self.pos_enc = PositionalEncoding(d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, d * 4, 0.1, batch_first=True,
                                       norm_first=True), enc_layers)
        self.embed = nn.Embedding(self.n_tokens, d, padding_idx=self.PAD)
        self.pos_dec = PositionalEncoding(d)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nhead, d * 4, 0.1, batch_first=True,
                                       norm_first=True), dec_layers)
        self.head = nn.Linear(d, self.n_tokens)
        self.d = d

    def encode(self, x):
        return self.encoder(self.pos_enc(self.backbone(x)))

    def forward(self, x, tgt_in):
        mem = self.encode(x)
        T = tgt_in.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.pos_dec(self.embed(tgt_in) * math.sqrt(self.d))
        out = self.decoder(h, mem, tgt_mask=mask,
                           tgt_key_padding_mask=(tgt_in == self.PAD))
        return self.head(out)

    @torch.no_grad()
    def generate(self, x, max_len=48):
        mem = self.encode(x)
        B = x.size(0)
        seq = torch.full((B, 1), self.BOS, device=x.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        for _ in range(max_len):
            T = seq.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            h = self.pos_dec(self.embed(seq) * math.sqrt(self.d))
            logits = self.head(self.decoder(h, mem, tgt_mask=mask))[:, -1]
            nxt = logits.argmax(-1)
            nxt[done] = self.PAD
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= (nxt == self.EOS)
            if done.all():
                break
        return seq[:, 1:]


def build(name, n_classes):
    if name == "crnn":
        return CRNN(n_classes)
    if name == "attn":
        return AttnOCR(n_classes)
    raise ValueError(name)
