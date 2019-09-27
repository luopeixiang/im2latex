import os
from os.path import join

import torch
from torch.nn.utils import clip_grad_norm_

from utils import cal_loss, cal_epsilon


class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 use_cuda=True, init_epoch=1, last_epoch=15):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch-1)*len(train_loader)
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"

        while self.epoch <= self.last_epoch:
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in self.train_loader:
                step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    print(mes.format(
                        self.epoch, self.step, len(self.train_loader),
                        100 * self.step / len(self.train_loader),
                        avg_loss,
                        2**avg_loss
                    ))
                    losses = 0.0

            # one epoch Finished, calcute val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()

        imgs = imgs.to(self.device)
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        epsilon = cal_epsilon(
            self.args.decay_k, self.total_step, self.args.sample_method)
        logits = self.model(imgs, tgt4training, epsilon)

        # calculate loss
        loss = cal_loss(logits, tgt4cal_loss)
        self.step += 1
        self.total_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)

                epsilon = cal_epsilon(
                    self.args.decay_k, self.total_step, self.args.sample_method)
                logits = self.model(imgs, tgt4training, epsilon)
                loss = cal_loss(logits, tgt4cal_loss)
                val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            print(mes.format(
                self.epoch, avg_loss, 2**avg_loss
            ))
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        return avg_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        print("Saving checkpoint to {}".format(save_path))

        # torch.save(self.model, model_path)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'args': self.args
        }, save_path)
