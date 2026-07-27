'''
Stage 2 improvement candidate, independent of models/SegRNNRocket.py:
injects top-K FFT-magnitude features of the raw lookback window into h_n
before decoding, instead of ROCKET's random-convolution features. Same
injection point, different (and more classically time-series-native)
feature source.

Diagnosis this strand tests: docs/stage2_new_strands_draft.md's Discussion
notes that three independent information sources (calendar timestamps,
attention-derived context, ROCKET features) injected into h_n have all
regressed -- this is a fourth, deliberately different source. Unlike
ROCKET (a classification-era technique adapted to regression), top-K FFT
magnitude is a direct, standard frequency-domain feature (docs/Pre-precessing.pdf's
feature-engineering slide covers window-based/frequency features generally).

Also meant to be run both with and without --power_transform 1
(data_provider/data_loader.py's Yeo-Johnson swap, models/SegRNN.py's
Strand 5): Yeo-Johnson is a nonlinear pointwise transform of the
amplitude, so it does not commute with the Fourier transform -- computing
FFT features on Yeo-Johnson-transformed values is a genuinely different
feature from computing them on raw/standard-scaled values, not a
redundant test. No model-level change is needed for that comparison --
--power_transform is a data_provider-level flag, independent of --model.

Change (only): a deterministic (no learnable/random parameters of its
own) top-K FFT-magnitude feature extractor runs over the raw
(last-value-normalized) window per channel; a trainable Linear projects
the K magnitudes to d_model and adds them to h_n:
h_n_aug = h_n + fft_proj(fft_features). Everything else (segment
partition, value embedding, GRU encoding, PMF decode, normalization) is
identical to models/SegRNN.py.

Only --dec_way pmf is supported (like SegRNNTime/SegRNNAttn/SegRNNRocket).
'''

import torch
import torch.nn as nn
from layers.RevIN import RevIN


class FFTFeatures(nn.Module):
    '''Deterministic top-K FFT-magnitude feature extractor. No learnable
    or random parameters -- unlike ROCKETFeatures, there is nothing to
    register as a buffer; the FFT itself is the whole "kernel bank".'''

    def __init__(self, seq_len, k=32):
        super().__init__()
        self.seq_len = seq_len
        self.k = k

    def forward(self, x):
        # x: (bc, 1, s) -> (bc, k)
        spec = torch.fft.rfft(x, dim=-1)         # bc, 1, s//2+1 (complex)
        mag = spec.abs().squeeze(1)              # bc, s//2+1
        k = min(self.k, mag.shape[-1])
        topk = torch.topk(mag, k, dim=-1).values  # bc, k
        if k < self.k:
            pad = torch.zeros(mag.shape[0], self.k - k, device=mag.device, dtype=mag.dtype)
            topk = torch.cat([topk, pad], dim=-1)
        return topk


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

        self.fft_k = getattr(configs, 'fft_k', 32)

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNFFT only supports --dec_way pmf (see module docstring)"

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

        # FFT feature bank + trainable projection -- the only structural addition
        self.fft = FFTFeatures(self.seq_len, k=self.fft_k)
        self.fft_proj = nn.Linear(self.fft_k, self.d_model)

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

        # top-K FFT-magnitude features of the raw (normalized) window, per channel
        fft_feats = self.fft(x.reshape(-1, 1, self.seq_len))  # bc, k
        fft_ctx = self.fft_proj(fft_feats)  # bc, d

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # bc,n,d   hn: 1,bc,d

        # inject FFT context into the encoder's final hidden state
        hn = hn + fft_ctx.unsqueeze(0)
        if self.rnn_type == "lstm":
            cn = cn + fft_ctx.unsqueeze(0)

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
