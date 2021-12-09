'''Utils function'''
import torch
import torch.nn as nn


def l1norm(X, dim=1, eps=1e-12):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def lambda_sigmoid(x, lamda=5):
    return 1 / ( 1 + torch.exp(-lamda * x) )

def min_max_resacle(x, eps=1e-12):
    return torch.div(x - x.min(), x.max() - x.min() + eps)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score








