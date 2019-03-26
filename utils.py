import pickle as pkl

import torch

from make_vocab import PAD_TOKEN, END_TOKEN, START_TOKEN, UNK_TOKEN


def collate_fn(sign2id, batch):
    # filter the pictures that have different weight or height
    size = batch[0][0].size()
    batch = [img_formula for img_formula in batch
             if img_formula[0].size() == size]
    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[1].split()),
               reverse=True)

    imgs, formulas = zip(*batch)
    formulas_tensor = formulas2tensor(formulas, sign2id)
    imgs = torch.stack(imgs, dim=0)

    bsize = len(batch)
    tgt4training = torch.cat(
        [torch.ones(bsize, 1).long()*START_TOKEN, formulas_tensor],
        dim=1
    )  # targets for training , begin with START_TOKEN
    tgt4cal_loss = torch.cat(
        [formulas_tensor, torch.ones(bsize, 1).long()*END_TOKEN],
        dim=1
    )  # targets for calculating loss , end with END_TOKEN
    return imgs, tgt4training, tgt4cal_loss


def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""
    formulas = [formula.split() for formula in formulas]
    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def count_parameters(model):
    """count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pkl.load(f)
    return vocab


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
