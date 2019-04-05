from os.path import join

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from make_vocab import PAD_TOKEN


class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.step = 0
        self.epoch = 1
        self.best_val_loss = 1e18

    def train(self):
        while self.epoch <= self.args.epochs:
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in self.train_loader:
                step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    total_step = len(self.train_loader)
                    print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
                        self.epoch, self.step, total_step,
                        100 * self.step / total_step,
                        losses / self.args.print_freq
                    ))
                    losses = 0.0
            # one epoch Finished, calcute val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()

        imgs = imgs.to(self.args.device)
        tgt4training = tgt4training.to(self.args.device)
        tgt4cal_loss = tgt4cal_loss.to(self.args.device)
        logits = self.model(imgs, tgt4training)

        # calculate loss
        loss = self.cal_loss(logits, tgt4cal_loss)
        self.step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                imgs = imgs.to(self.args.device)
                tgt4training = tgt4training.to(self.args.device)
                tgt4cal_loss = tgt4cal_loss.to(self.args.device)

                logits = self.model(imgs, tgt4training)
                loss = self.cal_loss(logits, tgt4cal_loss)
                val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            print("Epoch {}, validation average loss: {:.4f}".format(
                self.epoch, avg_loss
            ))
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model()
        return avg_loss

    def cal_loss(self, logits, targets):
        """args:
            logits: probability distribution return by model
                    [B, MAX_LEN, voc_size]
            targets: target formulas
                    [B, MAX_LEN]
        """
        padding = torch.ones_like(targets) * PAD_TOKEN
        mask = (targets != padding)

        targets = targets.masked_select(mask)
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, logits.size(2))
        ).contiguous().view(-1, logits.size(2))
        logits = torch.log(logits)

        assert logits.size(0) == targets.size(0)

        loss = F.nll_loss(logits, targets)
        return loss

    def save_model(self):
        print("Saving as best model...")
        torch.save(
            self.model.state_dict(),
            join(self.args.save_dir, 'best_ckpt')
        )
