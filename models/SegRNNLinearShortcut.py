'''
Stage 2 improvement candidate: blends SegRNN's recurrent path with a
DLinear-style direct linear path (Linear(seq_len -> pred_len), applied
per channel to the raw normalized window, bypassing the GRU entirely),
via a learned per-channel gate.

Why: the paper's own text (Section V-B2) notes that in the univariate
setting, "the lightweight DLinear outperforms the more complex
Transformer-based models such as PatchTST and iTransformer" -- simplicity
already wins inside the paper's own results, not just in this project's
Stage 2 findings (d_model=256, Huber loss). Blending a simple linear
shortcut alongside the RNN path -- rather than replacing the RNN with it,
or vice versa -- tests whether SegRNN's forecasts have systematic errors
a plain linear model doesn't share, letting the model learn how much to
trust each path per channel.

Change (only): a second, parallel Linear(seq_len -> pred_len) applied
directly to the normalized input (independent of the segment/GRU/PMF
path), blended with the RNN path's output via
y = gate*y_rnn + (1-gate)*y_linear, gate = sigmoid(per-channel learned
parameter, initialized at 0 so gate starts at 0.5 -- balanced, no prior
preference for either path). Both paths operate in the same normalized
space; only one denorm step is applied, after blending. Everything else
(segment partition, value embedding, GRU encoding, PMF decode,
normalization) is identical to models/SegRNN.py.

Only --dec_way pmf is supported (like SegRNNTime/SegRNNAttn/SegRNNRocket/SegRNNFFT).
'''

import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.rnn_type = configs.rnn_type
        self.dec_way = configs.dec_way
        self.seg_len = configs.seg_len
        self.channel_id = configs.channel_id
        self.revin = configs.revin

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNLinearShortcut only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # RNN path -- identical to models/SegRNN.py
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        if self.channel_id:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        # linear shortcut path + blend gate -- the only structural addition
        self.linear_shortcut = nn.Linear(self.seq_len, self.pred_len)
        self.blend_gate = nn.Parameter(torch.zeros(self.enc_in))  # sigmoid(0) = 0.5, balanced start

        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)

    def forward(self, x):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # -- RNN path --
        x_seg = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x_seg)
        else:
            _, hn = self.rnn(x_seg)  # hn: 1,bc,d

        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        if self.rnn_type == "lstm":
            _, (hy, cy) = self.rnn(pos_emb,
                                   (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                    cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
        else:
            _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y_rnn = self.predict(hy).view(-1, self.enc_in, self.pred_len)  # b,c,pred_len

        # -- linear shortcut path --
        y_linear = self.linear_shortcut(x)  # b,c,pred_len (Linear over the last dim, s -> pred_len)

        # -- blend --
        gate = torch.sigmoid(self.blend_gate).view(1, -1, 1)  # 1,c,1
        y = gate * y_rnn + (1 - gate) * y_linear

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
