'''
Stage 2 improvement candidate: retests models/SegRNNAttn.py's "remove the
encoder bottleneck" hypothesis at (essentially) zero added parameters,
to separate two possible explanations for why attention regressed
(docs/stage2_revin_attn_draft.md strand 4, worse at every horizon):
(a) giving the decoder access to more than just h_n doesn't help this
data/architecture, or (b) it could help, but SegRNNAttn's learned
Q/K/V projections added enough capacity to repeat the same "more
capacity hurts" pattern every other Stage 2 strand has shown.

Change (only): the encoder keeps its full per-segment output sequence
(all n states, not just h_n), exactly like SegRNNAttn -- but instead of
learned attention, a parameter-free mean-pool (--pool_type mean, or max
via --pool_type max) collapses it to a single context vector per (b,c),
added directly to h_n with no learned projection at all:
h_n_aug = h_n + pool(enc_outputs). This adds precisely zero new
parameters versus models/SegRNN.py -- mean/max pooling has none, unlike
SegRNNAttn's attn_query/key/value or SegRNNRocket/SegRNNFFT's projection
layers. If this also regresses, it points at (a); if it doesn't, it
points at (b) and reopens the bottleneck-removal idea as worth pursuing
with a lighter mechanism than full attention.

Only --dec_way pmf is supported (like SegRNNAttn/SegRNNRocket/SegRNNFFT).
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

        self.pool_type = getattr(configs, 'pool_type', 'mean')
        assert self.pool_type in ['mean', 'max']

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNPoolContext only supports --dec_way pmf (see module docstring)"

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

        # no learnable parameters here -- pooling is the entire "mechanism"

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

        # encoding -- capture the full per-segment output sequence, not just h_n
        if self.rnn_type == "lstm":
            enc_outputs, (hn, cn) = self.rnn(x)
        else:
            enc_outputs, hn = self.rnn(x)  # enc_outputs: bc,n,d   hn: 1,bc,d

        # parameter-free pooling context over all n encoder segment states
        if self.pool_type == 'mean':
            context = enc_outputs.mean(dim=1)  # bc, d
        else:
            context = enc_outputs.max(dim=1).values  # bc, d

        hn = hn + context.unsqueeze(0)
        if self.rnn_type == "lstm":
            cn = cn + context.unsqueeze(0)

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
