'''
Stage 2 improvement candidate: replaces the value-embedding step
(Linear(seg_len -> d_model) + ReLU) with a small 1D convolution over each
segment (kernel width < seg_len, so it slides within a segment) followed
by mean-pooling over time, instead of one fully-connected layer applied
to the whole segment at once.

Why this is a different change from Linear, not a relabeling of it: a
Conv1d with kernel_size == seg_len and stride == seg_len (one application
per segment, no sliding) would be mathematically equivalent to the
original Linear -- no new inductive bias. This strand instead uses a
kernel width smaller than seg_len (--conv_kernel_size, default 5), so the
conv genuinely slides across multiple positions within each segment,
giving the embedding step a translation-equivariant, local-pattern
inductive bias that a single Linear layer does not have (Linear treats
the w values as a flat feature vector; each input position gets its own
independent weight column, with no assumption that nearby timesteps
behave similarly).

CLAUDE.md flags this as the higher-risk of its two listed architecture
candidates, citing ISMRNN (arXiv 2407.10768) reporting conv hurting on
all datasets except Weather -- tested anyway as a real data point, going
in with that prior rather than a blank slate.

Change (only): self.valueEmbedding is a Conv1d+ReLU applied within each
segment (padding='same', so segment length is preserved) followed by
mean-pooling over the segment's time dimension, producing one d_model
vector per segment -- the same shape the original Linear+ReLU produced,
so nothing downstream changes. Everything else (segment partition, GRU
encoding, PMF decode, positional embedding, normalization) is identical
to models/SegRNN.py.

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

        self.conv_kernel_size = getattr(configs, 'conv_kernel_size', 5)

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNConvEmbed only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # value embedding -- conv within each segment instead of one Linear
        # over the whole segment (the only structural change)
        self.valueEmbeddingConv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.d_model, kernel_size=self.conv_kernel_size,
                      padding=self.conv_kernel_size // 2),
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

        # segment    b,c,s -> bc,n,w
        x_seg = x.reshape(-1, self.seg_num_x, self.seg_len)

        # conv embedding: treat each segment as its own length-w, single-channel
        # signal -> bcn,1,w -> bcn,d,w -> mean-pool over w -> bcn,d -> bc,n,d
        x_conv_in = x_seg.reshape(-1, 1, self.seg_len)
        conv_out = self.valueEmbeddingConv(x_conv_in)     # bcn, d, w
        seg_embed = conv_out.mean(dim=-1)                  # bcn, d
        x = seg_embed.reshape(-1, self.seg_num_x, self.d_model)  # bc, n, d

        # encoding -- identical to models/SegRNN.py from here on
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # hn: 1,bc,d

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
