"""LMDB-backed dataset, curriculum sampler, and augmentation."""
import io
import json
import random

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class Charset:
    """Maps characters <-> indices. Index 0 is reserved for CTC blank / pad."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank = 0

    def __len__(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode_ctc(self, ids):
        """Collapse repeats, drop blanks -- standard CTC greedy decode."""
        out, prev = [], None
        for i in ids:
            if i != self.blank and i != prev:
                out.append(self.itos.get(i, ""))
            prev = i
        return "".join(out)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["charset"])


class Augment:
    """Applied ONLY to synthetic training data.

    Your corpus is ~99% font-rendered. Without augmentation a model reaches
    near-zero error on synthetic validation and then collapses on real photos:
    it has learned the renderer, not the script. These transforms inject the
    degradations real images have -- blur, noise, compression, contrast shift.
    """

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        if random.random() < 0.4:                       # gaussian blur
            from PIL import ImageFilter
            img = Image.fromarray(arr.astype(np.uint8))
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
            arr = np.array(img).astype(np.float32)
        if random.random() < 0.4:                       # sensor noise
            arr += np.random.normal(0, random.uniform(3, 12), arr.shape)
        if random.random() < 0.4:                       # contrast / brightness
            arr = (arr - 128) * random.uniform(0.6, 1.4) + 128 + random.uniform(-25, 25)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if random.random() < 0.3:                       # JPEG artifacts
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=random.randint(30, 75))
            arr = np.array(Image.open(buf).convert("L"))
        return Image.fromarray(arr)


class LMDBDataset(Dataset):
    def __init__(self, paths, charset, height=64, max_width=512, augment=False):
        self.paths = [str(p) for p in paths]
        self.charset = charset
        self.height = height
        self.max_width = max_width
        self.aug = Augment() if augment else None
        self.envs, self.index, self.tiers = None, [], []
        for pi, p in enumerate(self.paths):
            env = lmdb.open(p, readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                n = int(txn.get(b"num-samples"))
                for i in range(n):
                    self.index.append((pi, i))
                    self.tiers.append(txn.get(f"tier-{i:09d}".encode()).decode())
            env.close()

    def _env(self, pi):
        if self.envs is None:   # opened lazily, per worker process
            self.envs = [lmdb.open(p, readonly=True, lock=False, readahead=False)
                         for p in self.paths]
        return self.envs[pi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pi, i = self.index[idx]
        with self._env(pi).begin() as txn:
            img = Image.open(io.BytesIO(txn.get(f"img-{i:09d}".encode())))
            label = txn.get(f"lbl-{i:09d}".encode()).decode()
        img = img.convert("L")
        if self.aug:
            img = self.aug(img)
        w, h = img.size
        new_w = min(self.max_width, max(8, int(w * self.height / h)))
        img = img.resize((new_w, self.height), Image.BILINEAR)
        arr = (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(arr)[None], torch.tensor(self.charset.encode(label)), label


def collate(batch):
    """Right-pad images to the widest in the batch; pack labels for CTC."""
    imgs, targets, texts = zip(*batch)
    max_w = max(i.shape[-1] for i in imgs)
    padded = torch.zeros(len(imgs), 1, imgs[0].shape[1], max_w)
    widths = []
    for i, im in enumerate(imgs):
        padded[i, :, :, :im.shape[-1]] = im
        widths.append(im.shape[-1])
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return (padded, torch.cat(targets) if len(targets) else torch.tensor([]),
            lengths, torch.tensor(widths), list(texts))


class CurriculumSampler(Sampler):
    """One model, staged data mix -- NOT three separate models.

    Character, word and sentence are the same task over the same output
    alphabet; a character image is just a length-1 sequence. So we never swap
    models, we only reweight what gets sampled.

    Earlier tiers never fall to zero. Dropping them entirely causes
    catastrophic forgetting: the model trades glyph precision for language
    modelling and short-sequence accuracy degrades.
    """

    STAGES = [
        (0,  {"char": 0.60, "word": 0.35, "sentence": 0.05}),
        (3,  {"char": 0.25, "word": 0.45, "sentence": 0.30}),
        (8,  {"char": 0.10, "word": 0.30, "sentence": 0.60}),
    ]

    def __init__(self, tiers, epoch=0, samples_per_epoch=None):
        self.by_tier = {"char": [], "word": [], "sentence": []}
        for i, t in enumerate(tiers):
            self.by_tier[t].append(i)
        self.n = samples_per_epoch or len(tiers)
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.weights = self.STAGES[0][1]
        for start, w in self.STAGES:
            if epoch >= start:
                self.weights = w

    def __iter__(self):
        out = []
        for tier, frac in self.weights.items():
            pool = self.by_tier[tier]
            if not pool:
                continue
            k = int(self.n * frac)
            # sample with replacement: char pool is far smaller than sentence
            out += [random.choice(pool) for _ in range(k)]
        random.shuffle(out)
        return iter(out)

    def __len__(self):
        return self.n
