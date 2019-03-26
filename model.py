
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

INIT = 1e-2


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size,
                 enc_rnn_h, dec_rnn_h, n_layer=1):
        super(Im2LatexModel, self).__init__()

        # follow the original paper's table2: CNN specification
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), (0, 0)),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1))
        )

        self.rnn_encoder = nn.LSTM(64, enc_rnn_h,
                                   bidirectional=True,
                                   batch_first=True)
        self.rnn_decoder = nn.LSTMCell(enc_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        # enc_rnn_h*2 is the dimension of context
        self.W_c = nn.Linear(dec_rnn_h+2*enc_rnn_h, enc_rnn_h)
        self.W_out = nn.Linear(enc_rnn_h, out_size)

        # a trainable initial hidden state V_h_0 for each row
        self.V_h_0 = nn.Parameter(torch.Tensor(n_layer*2, enc_rnn_h))
        self.V_c_0 = nn.Parameter(torch.Tensor(n_layer*2, enc_rnn_h))
        init.uniform_(self.V_h_0, -INIT, INIT)
        init.uniform_(self.V_c_0, -INIT, INIT)

        # Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(dec_rnn_h))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_h = nn.Linear(dec_rnn_h, dec_rnn_h)
        self.W_v = nn.Linear(enc_rnn_h*2, dec_rnn_h)

    def forward(self, imgs, formulas):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]

        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        row_enc_out, hiddens = self.encode(imgs)
        # init decoder's states
        dec_states, O_t = self.init_decoder(row_enc_out, hiddens)
        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(
                dec_states, O_t, row_enc_out, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 64, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 64]

        # Prepare data for Row Encoder
        # poccess data like a new big batch
        B, H, W, out_channels = encoded_imgs.size()
        encoded_imgs = encoded_imgs.contiguous().view(B*H, W, out_channels)
        # prepare init hidden for each row
        init_hidden_h = self.V_h_0.unsqueeze(
            1).expand(-1, B*H, -1).contiguous()
        init_hidden_c = self.V_c_0.unsqueeze(
            1).expand(-1, B*H, -1).contiguous()
        init_hidden = (init_hidden_h, init_hidden_c)

        # Row Encoder
        row_enc_out, (h, c) = self.rnn_encoder(encoded_imgs, init_hidden)
        # row_enc_out [B*H, W, enc_rnn_h]
        # hidden: [2, B*H, enc_rnn_h]
        row_enc_out = row_enc_out.view(B, H, W, -1)  # [B, H, W, enc_rnn_h]
        h, c = h.view(2, B, H, -1), c.view(2, B, H, -1)

        return row_enc_out, (h, c)

    def step_decoding(self, dec_states, O_t, enc_out, tgt):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, O_t], dim=1)  # [B, emb_size+enc_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)

        context_t, attn_scores = self._get_attn(enc_out, dec_states[0])
        # [B, enc_rnn_h]
        O_t = self.W_c(torch.cat([h_t, context_t], dim=1)).tanh()

        # calculate logit
        logit = F.softmax(self.W_out(O_t), dim=1)  # [B, out_size]

        return (h_t, c_t), O_t, logit

    def _get_attn(self, enc_out, prev_h):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, H, W, enc_rnn_h]
            prev_h: the previous time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, enc_rnn_h]
            attn_scores: Attention scores
        """
        # self.W_v(enc_out) [B, H, W, enc_rnn_h]
        # self.W_h(prev_h) [B, enc_rnn_h]
        B, H, W, _ = enc_out.size()
        linear_prev_h = self.W_h(prev_h).view(B, 1, 1, -1)
        linear_prev_h = linear_prev_h.expand(-1, H, W, -1)
        e = torch.sum(
            self.beta * torch.tanh(
                linear_prev_h +
                self.W_v(enc_out)
            ),
            dim=-1
        )  # [B, H, W]

        alpha = F.softmax(e.view(B, -1), dim=-1).view(B, H, W)
        attn_scores = alpha.unsqueeze(-1)
        context = torch.sum(attn_scores * enc_out,
                            dim=[1, 2])  # [B, enc_rnn_h]

        return context, attn_scores

    def init_decoder(self, enc_out, hiddens):
        """args:
            enc_out: the output of row encoder [B, H, W, enc_rnn_h]
            hidden: the last step hidden of row encoder [2, B, H, enc_rnn_h]
          return:
            h_0, c_0  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, enc_rnn_h]
            for decoder
        """

        h, c = hiddens
        h, c = self._convert_hidden(h), self._convert_hidden(c)
        context_0 = enc_out.mean(dim=[1, 2])
        init_O = torch.tanh(
            self.W_c(torch.cat([h, context_0], dim=1))
        )
        return (h, c), init_O

    def _convert_hidden(self, hidden):
        """convert row encoder hidden to decoder initial hidden"""
        hidden = hidden.permute(1, 2, 0, 3).contiguous()
        # Note that 2*enc_rnn_h = dec_rnn_h
        hidden = hidden.view(hidden.size(
            0), hidden.size(1), -1)  # [B, H, dec_rnn_h]
        hidden = hidden.mean(dim=1)  # [B, dec_rnn_h]

        return hidden
