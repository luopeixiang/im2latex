
from functools import partial

import torch
from torch.utils.data import DataLoader
#test data loader
from utils import collate_fn
from data import Im2LatexDataset
from make_vocab import make_vocab
from model import Im2LatexModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    path = "/home/luo/Github_Project/Reference/im2markup/data/sample"
    vocab = make_vocab(path)
    loader = DataLoader(Im2LatexDataset(path, 'test'),
                   batch_size=2,
                   collate_fn=partial(collate_fn, vocab.sign2id))

    #测试一下模型能不能运行
    imgs, formulas, _ = loader.__iter__().__next__()

    out_size = len(vocab)
    model = Im2LatexModel(out_size, 80, 256, 512)
    num_params = count_parameters(model)
    with torch.no_grad():
        logits = model(imgs, formulas)
    print(logits.shape)
    print(logits)
    print("共有 ", num_params, "个参数")

main()
