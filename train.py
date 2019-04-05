import argparse
from functools import partial
from os.path import join

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import collate_fn, build_vocab
from data import Im2LatexDataset
from model import Im2LatexModel
from training import Trainer
from make_vocab import make_vocab


def main():
    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    parser.add_argument('--path', required=True, help='root of the model')

    # model args
    parser.add_argument(
        "--emb_dim", type=int, default=80, help="Embedding size")
    parser.add_argument(
        "--enc_rnn_h",
        type=int,
        default=256,
        help="The hidden state of the encoder RNN")
    parser.add_argument(
        "--dec_rnn_h",
        type=int,
        default=512,
        help="The hidden state of the decoder RNN")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./sample_data/",
        help="The dataset's dir")
    # training args
    parser.add_argument(
        "--cuda", action='store_true', default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="Learning Rate Decay Rate")
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Learning Rate Decay Patience")
    parser.add_argument(
        "--clip", type=float, default=5.0, help="The max gradient norm")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./ckpt",
        help="The dir to save checkpoint")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./sample_data/vocab.pkl",
        help="The path to vocab file")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=4,
        help="The frequency to print message")

    args = parser.parse_args()

    # Building vocab
    make_vocab(args.data_path)
    vocab = build_vocab(join(args.data_path, 'vocab.pkl'))

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    args.__dict__['device'] = device

    # data loader
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'train'),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        #pin_memory=True if use_cuda else False,
        num_workers=4)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate'),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        #pin_memory=True if use_cuda else False,
        num_workers=4)

    # construct model
    vocab_size = len(vocab)
    model = Im2LatexModel(vocab_size, args.emb_dim, args.enc_rnn_h,
                          args.dec_rnn_h)
    model = model.to(device)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True)

    # init trainer
    trainer = Trainer(optimizer, model, lr_scheduler, train_loader, val_loader,
                      args)
    # begin training
    trainer.train()


if __name__ == "__main__":
    main()
