'''
Stage 2 improvement candidate: revised placement of models/SegRNNLayerNorm.py's
idea, which regressed (docs/stage2_followup_ideas_draft.md strand 19).
That version normalized right after the input embedding (Linear+ReLU),
which may double up on the existing last-value normalization or disrupt
a scale the GRU already expects at its input. This version instead
normalizes h_n -- the actual bottleneck representation -- right before
decoding, the same way a Transformer block normalizes hidden states
rather than raw inputs.

Change (only): nn.LayerNorm(d_model) applied to hn immediately after
encoding, before it's used to initialize the decode step. Everything
else (segment partition, value embedding, GRU encoding, PMF decode,
positional embedding, normalization) is identical to models/SegRNN.py.
Adds 2*d_model parameters, same near-zero cost as the original LayerNorm
strand -- this is a change in placement, not in cost.

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
        assert self.dec_way == 'pmf', "SegRNNLayerNormHidden only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- identical to models/SegRNN.py
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

        # normalizes h_n itself -- the only structural change from models/SegRNN.py
        self.hn_norm = nn.LayerNorm(self.d_model)
        if self.rnn_type == "lstm":
            self.cn_norm = nn.LayerNorm(self.d_model)

        if self.channel_id:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
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

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
            cn = self.cn_norm(cn)
        else:
            _, hn = self.rnn(x)  # hn: 1,bc,d
        hn = self.hn_norm(hn)

        # decoding (pmf only, see the dec_way assert above)
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
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
