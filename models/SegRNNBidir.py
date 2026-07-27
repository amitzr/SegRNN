'''
Stage 2 improvement candidate: bidirectional GRU encoder (CLAUDE.md's own
unimplemented improvement B). The whole look-back window is past data by
the time encoding happens, so processing it in both directions is not
leakage -- the model still only ever sees x when predicting y.

Structural consequence worth being upfront about: SegRNN's efficiency
design deliberately shares one GRU cell between encode and decode. A
bidirectional encoder can't be reused directly for the PMF decode step
(decoding is a single forward step from a starting hidden state -- there
is no "reverse direction" for it to run over), so this strand necessarily
breaks that sharing and adds a second, decode-only unidirectional GRU
cell. That is a real, deliberate deviation from SegRNN's efficiency
philosophy: this strand roughly doubles encoder parameters (bidirectional
GRU) *and* adds a full second GRU's worth of decode-only parameters, on
top of a small projection layer. Given docs/stage2_new_strands_draft.md's
established pattern (every added-capacity strand so far has regressed,
and AIC/BIC formally rejects even d_model=512's parameter count), this is
the single heaviest strand tested in the project -- go in expecting it to
need a strong effect to overcome that much added capacity, not a free win.

Change: encoder is nn.*RNN(..., bidirectional=True) with num_layers=1.
Its final forward/backward hidden states (2, bc, d) are concatenated (bc,
2d) and projected back to d via a trainable Linear (bidir_proj), matching
CLAUDE.md's own description. Decoding uses a separate, unidirectional GRU
cell (decode_rnn) initialized from the projected state -- everything else
(segment partition, value embedding, positional embedding, predict head,
normalization) is identical to models/SegRNN.py.

Only --dec_way pmf is supported, and only rnn_type='gru' (a bidirectional
LSTM/RNN encoder feeding a unidirectional GRU decoder is not implemented
here -- keep --rnn_type gru, the paper's own best-performing cell, Fig. 8).
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

        assert self.rnn_type == 'gru', "SegRNNBidir only supports --rnn_type gru (see module docstring)"
        assert self.dec_way == 'pmf', "SegRNNBidir only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- bidirectional (the only structural addition on the encode side)
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.encoder_rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                                   batch_first=True, bidirectional=True)
        self.bidir_proj = nn.Linear(2 * self.d_model, self.d_model)

        # decoder -- separate, unidirectional cell (weight-sharing broken, see docstring)
        self.decode_rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
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

        # bidirectional encoding: hn: 2,bc,d (dim0: forward, backward)
        _, hn = self.encoder_rnn(x)
        hn_cat = torch.cat([hn[0], hn[1]], dim=-1)  # bc, 2d
        hn_proj = self.bidir_proj(hn_cat)           # bc, d
        hn_proj = hn_proj.unsqueeze(0)               # 1,bc,d -- decode_rnn's initial state

        # decoding (pmf only, see the dec_way assert above)
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        _, hy = self.decode_rnn(pos_emb, hn_proj.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
