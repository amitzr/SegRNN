'''
Stage 2 improvement candidate: injects multi-scale window-based
statistical features into h_n before decoding -- the same injection
point as models/SegRNNRocket.py and models/SegRNNFFT.py, a different
(and more classical) feature source. "Window-based features" is its own
named item in docs/Pre-precessing.pdf's feature-engineering taxonomy,
distinct from the window/convolutional/ROCKET/shapelet items already
tried (ROCKET: models/SegRNNRocket.py; FFT, the taxonomy's frequency-
domain analogue: models/SegRNNFFT.py) -- this is the simplest, most
classical item in that same taxonomy, untested until now.

Feature set: mean, standard deviation, and linear-trend slope, each
computed over the last {24, 168, 720} timesteps of the raw
(last-value-normalized) look-back window -- day, week, and the full
look-back, matching ETTh1's own seg_len=24 and its known daily/weekly
periodicity (not arbitrary window sizes). 3 statistics x 3 windows = 9
features per channel, all deterministic (no learnable or random
parameters of their own, like FFTFeatures) -- a trainable Linear
projects them to d_model and adds them to h_n:
h_n_aug = h_n + window_proj(window_features).

Why this is a fair, independent test of the "inject into h_n" pattern:
three prior information sources injected this way (calendar timestamps,
attention-derived context, ROCKET/FFT features) have all regressed
(docs/stage2_report_draft.md, docs/stage2_revin_attn_draft.md,
docs/stage2_new_strands_draft.md, docs/stage2_architecture_variants_draft.md).
This is a fourth, classical, and (unlike ROCKET) non-adapted-from-
classification feature source, testing whether that pattern is about the
injection mechanism itself or specific to the feature sources tried so
far.

Only --dec_way pmf is supported (like SegRNNTime/SegRNNAttn/SegRNNRocket/SegRNNFFT).
'''

import torch
import torch.nn as nn
from layers.RevIN import RevIN


class WindowFeatures(nn.Module):
    '''Deterministic multi-scale window statistics -- mean, std, and
    linear-trend slope over each of window_sizes. No learnable or random
    parameters, like FFTFeatures; unlike FFTFeatures, these are the
    classical descriptive statistics docs/Pre-precessing.pdf names as
    "window-based features," not a signal-processing transform.'''

    def __init__(self, window_sizes=(24, 168, 720)):
        super().__init__()
        self.window_sizes = window_sizes

    def forward(self, x):
        # x: (bc, 1, s) -> (bc, 3*len(window_sizes))
        feats = []
        for w in self.window_sizes:
            xw = x[..., -w:].squeeze(1)  # bc, w
            mean = xw.mean(dim=-1)
            std = xw.std(dim=-1)
            t = torch.arange(w, device=x.device, dtype=x.dtype)
            t = t - t.mean()
            denom = (t * t).sum().clamp_min(1e-8)
            slope = (xw * t.unsqueeze(0)).sum(dim=-1) / denom
            feats += [mean, std, slope]
        return torch.stack(feats, dim=-1)


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

        self.window_sizes = (24, 168, 720)

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNWindowStats only supports --dec_way pmf (see module docstring)"

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

        # window-statistics feature bank + trainable projection -- the only structural addition
        self.window_features = WindowFeatures(self.window_sizes)
        self.window_proj = nn.Linear(3 * len(self.window_sizes), self.d_model)

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

        # multi-scale window statistics of the raw (normalized) window, per channel
        window_feats = self.window_features(x.reshape(-1, 1, self.seq_len))  # bc, 9
        window_ctx = self.window_proj(window_feats)  # bc, d

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # hn: 1,bc,d

        # inject window-statistics context into the encoder's final hidden state
        hn = hn + window_ctx.unsqueeze(0)
        if self.rnn_type == "lstm":
            cn = cn + window_ctx.unsqueeze(0)

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
