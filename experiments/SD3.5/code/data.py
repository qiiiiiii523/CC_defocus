"""Paired CNSeg loader using the repository's fixed JSONL manifests."""
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairedManifest(Dataset):
    def __init__(self, manifest, repo_root, size=256, train=True):
        self.root = Path(repo_root)
        self.rows = [json.loads(x) for x in Path(manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
        self.size, self.train = size, train

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        blur = Image.open(self.root / r['blur_path']).convert('RGB')
        clear = Image.open(self.root / r['clear_path']).convert('RGB')
        blur = TF.resize(blur, [self.size, self.size], antialias=True)
        clear = TF.resize(clear, [self.size, self.size], antialias=True)
        if self.train and torch.rand(()) < 0.5:
            blur, clear = TF.hflip(blur), TF.hflip(clear)
        return {'blur': TF.to_tensor(blur) * 2 - 1, 'clear': TF.to_tensor(clear) * 2 - 1,
                'sample_id': r['sample_id']}
