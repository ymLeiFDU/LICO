import torch
import torch.nn.functional as F
import numpy as np


def sim_matrix_pre(labels, text_tensors, temp, token_fc, noise = False):

    labels = labels.type(torch.cuda.FloatTensor)
    batch_text_tensors = torch.cuda.FloatTensor(labels.size(0), text_tensors.size(1))
    for i in range(labels.size(0)):
        class_i = int(labels[i].data.cpu().numpy())
        batch_text_tensors[i, :] = text_tensors[class_i, :]

    if token_fc is not None:
        batch_text_tensors = token_fc(batch_text_tensors)
    # temp = 1
    numerator = pdists(batch_text_tensors, squared = False, _type = 'euc', noise = noise)
    matrix = F.softmax(numerator / temp, dim = 1)

    return matrix, batch_text_tensors


def pdists(A, squared = False, _type = 'euc', noise = False):

    A = F.normalize(A, dim = 1)
    if noise == True:
        A = add_noise(A)

    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)

    if _type == 'euc':
        res = (norm + norm.t() - 2 * prod)
        if noise == True:
            res = add_noise(res)
    elif _type == 'cos':
        res = 1-(prod / norm)
        if noise == True:
            res = add_noise(res)

    eps = 1e-4
    if squared:
        res.diag == 0
        return res.clamp(min = eps)
    else:
        res = res.clamp(min = eps).sqrt()
        return res







