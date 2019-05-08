import torch
import numpy as np
import torch.nn.functional as F

# def pair_similarity(x, y):
#     '''
#     x: n * dx
#     y: m * dy
#     '''
#
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     ps = torch.eq(x,y).squeeze(2)
#     ps[ps==0] = -1
#     return ps

def cdist(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist

def pair_similarity(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    ps = torch.eq(x,y).squeeze(2)
    return ps


def relation_loss(relation_score, labelS):
    loss = torch.nn.MSELoss(reduction='none')(relation_score, labelS.float()).sum() / np.sqrt(labelS.size(0))
    return loss

