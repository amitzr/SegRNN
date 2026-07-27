'''
Stage 2 improvement candidate: stacks the encoder GRU to num_layers=2
(--encoder_layers, default 2) instead of the original single layer.

This is a different axis from the paper's own segment-length ablation
(Fig. 6, docs/segrnn_project_plan.md): that ablation varies n = L/w, the
number of sequential *recurrent steps* a single-layer GRU unrolls over,
and the paper already found and picked the sweet spot on that axis
(w=48, or w=24 for ETTh1's own script). This strand instead stacks
additional GRU *layers* processing the same n-step sequence -- untested
by the paper and untested elsewhere in this project.

Given the pattern across every other Stage 2 strand so far
(docs/stage2_new_strands_draft.md's Discussion: every strand that added
capacity or richness regressed; only reducing d_model helped, and AIC/BIC
rejected d_model=512 outright), the honest prior going in is skepticism --
a second GRU layer meaningfully increases encoder parameter count, the
same direction AIC/BIC just formally rejected. Tested anyway as a genuine
data point on the "does more capacity ever help" question, since depth
and width are architecturally different levers.

Change (only): self.rnn's num_layers is configurable (encoder_layers,
default 2) instead of hardcoded to 1. The PMF decode step's hidden-state
reshape (previously hardcoded to a leading dim of 1, since num_layers=1
was implicit) is generalized to self.encoder_layers -- everything else
(segment partition, value embedding, positional embedding, predict head,
normalization) is identical to models/SegRNN.py. The decode step still
uses the *same* shared GRU cell for encode and decode (unlike
SegRNNBidir.py, where a stacked *bidirectional* encoder breaks that
sharing) -- nn.GRU's forward pass works the same way regardless of
num_layers, so the weight-sharing trick still applies here.

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

        self.encoder_layers = getattr(configs, 'encoder_layers', 2)

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNDeepEncoder only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- identical to models/SegRNN.py except num_layers
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.encoder_layers,
                              bias=True, batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.encoder_layers,
                              bias=True, batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.encoder_layers,
                              bias=True, batch_first=True, bidirectional=False)

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
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y L:encoder_layers
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding -- hn/cn now have leading dim encoder_layers, not 1
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)  # bc,n,d   hn/cn: L,bc,d
        else:
            _, hn = self.rnn(x)  # hn: L,bc,d

        # decoding (pmf only, see the dec_way assert above)
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        # hn/cn: L,bc,d -> L,bc,md -> L,bcm,d (generalizes the original's
        # hardcoded leading dim of 1 to encoder_layers)
        if self.rnn_type == "lstm":
            _, (hy, cy) = self.rnn(pos_emb,
                                   (hn.repeat(1, 1, self.seg_num_y).view(self.encoder_layers, -1, self.d_model),
                                    cn.repeat(1, 1, self.seg_num_y).view(self.encoder_layers, -1, self.d_model)))
        else:
            _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(self.encoder_layers, -1, self.d_model))
        # hy: L,bcm,d -- only the last layer's state is what the shared GRU
        # cell actually used as its "current" hidden state for the (single-step)
        # decode output; nn.GRU/RNN/LSTM's output sequence already reflects
        # this internally, so predict() reads hy[-1:] as in the original code's
        # implicit L=1 case.
        y = self.predict(hy[-1:]).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
