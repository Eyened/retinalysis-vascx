from ast import List, Tuple
import torch
from torchmetrics.functional import dice

from vascx.retina import Retina

def pairwise_f1(pairs):
    return [f1(p[0], p[1]) for p in pairs]

def f1(r1: Retina, r2: Retina):
    scores = {}

    assert len(r1.layers) == len(r2.layers)
    for key in r1.layers.keys():
        assert key in r2.layers
        scores[key] = dice(torch.tensor(r1.layers[key].binary, dtype=torch.uint8), torch.tensor(r2.layers[key].binary, dtype=torch.uint8), num_classes=2, average=None)

    return scores