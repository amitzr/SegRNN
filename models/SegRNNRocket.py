'''
Stage 2 improvement candidate, independent of the calendar-feature
(models/SegRNNTime.py) and attention (models/SegRNNAttn.py) strands: adds
ROCKET-style random-convolutional features of the raw lookback window as
extra information injected into h_n before decoding.

Source: docs/Pre-precessing.pdf's feature-engineering slide lists ROCKET
(random convolutional kernels, Max + PPV pooling) as a window-based feature
technique. ROCKET is a classification-era technique (sktime's
transformations.panel.rocket.Rocket) -- there is no textbook precedent for
using it inside a forecaster, so this is a genuine, honestly-labelled
adaptation, not a documented method. To avoid an extra sktime dependency on
Colab, the kernels are implemented directly in PyTorch: fixed (untrained,
buffer-registered) 1D convolutions with random weights/dilations/bias, one
Max and one PPV (proportion of positive values) feature per kernel -- the
same two pooling statistics the ROCKET paper uses.

Diagnosis this strand tests: docs/stage2_report_draft.md's three
calendar-feature attempts all injected extra information into h_n and all
regressed (docs/stage2_revin_attn_draft.md's Discussion: "every attempt to
give the model more...made results worse"). This strand deliberately repeats
that same injection pattern with a different, non-calendar information
source (features of the raw series itself, not timestamps) as an honest,
independent check of whether that finding is specific to calendar
information or a more general property of this architecture/dataset.

Change (only): after the encoder produces h_n exactly as in
models/SegRNN.py, a fixed random-kernel bank runs over the (already
last-value-normalized) raw window per channel, producing a
2*num_kernels-dim feature vector (Max + PPV per kernel), which a trainable
Linear layer projects to d_model and adds to h_n:
h_n_aug = h_n + rocket_proj(rocket_features). Everything else (segment
partition, value embedding, GRU encoding, PMF decode, normalization) is
unchanged from models/SegRNN.py.

Only --dec_way pmf is supported (like SegRNNTime/SegRNNAttn).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class ROCKETFeatures(nn.Module):
    '''Fixed (untrained) random-convolution feature bank: Max + PPV per
    kernel, the two pooling statistics from the ROCKET paper. Kernels are
    registered as buffers (not parameters), so they never receive gradients
    and add zero trainable parameters beyond the projection layer that
    consumes their output.'''

    def __init__(self, num_kernels=32, kernel_size=9, num_dilations=4, seed=2024):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(num_kernels, 1, kernel_size, generator=g)
        weight = weight - weight.mean(dim=-1, keepdim=True)  # ROCKET: mean-centered kernels
        bias = torch.rand(num_kernels, generator=g) * 2 - 1  # ROCKET: bias ~ U(-1, 1)
        dilations = 2 ** torch.randint(0, num_dilations, (num_kernels,), generator=g)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.register_buffer('dilations', dilations)
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

    def forward(self, x):
        # x: (bc, 1, s) -> (bc, 2*num_kernels)
        feats = []
        for k in range(self.num_kernels):
            d = int(self.dilations[k].item())
            pad = ((self.kernel_size - 1) * d) // 2
            out = F.conv1d(x, self.weight[k:k + 1], bias=self.bias[k:k + 1], dilation=d, padding=pad)
            feats.append(out.amax(dim=-1))          # Max
            feats.append((out > 0).float().mean(dim=-1))  # PPV
        return torch.cat(feats, dim=-1)


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

        self.rocket_kernels = getattr(configs, 'rocket_kernels', 32)
        self.rocket_kernel_size = getattr(configs, 'rocket_kernel_size', 9)

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNRocket only supports --dec_way pmf (see module docstring)"

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

        # ROCKET feature bank + trainable projection -- the only structural addition
        self.rocket = ROCKETFeatures(num_kernels=self.rocket_kernels, kernel_size=self.rocket_kernel_size)
        self.rocket_proj = nn.Linear(2 * self.rocket_kernels, self.d_model)

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

        # ROCKET features of the raw (normalized) window, per channel
        rocket_feats = self.rocket(x.reshape(-1, 1, self.seq_len))  # bc, 2*num_kernels
        rocket_ctx = self.rocket_proj(rocket_feats)  # bc, d

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # bc,n,d   hn: 1,bc,d

        # inject ROCKET context into the encoder's final hidden state
        hn = hn + rocket_ctx.unsqueeze(0)
        if self.rnn_type == "lstm":
            cn = cn + rocket_ctx.unsqueeze(0)

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
