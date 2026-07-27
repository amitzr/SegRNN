'''
Stage 2 improvement candidate: ties the value-embedding weight
(Linear(seg_len -> d_model)) and the predict head's weight
(Linear(d_model -> seg_len)) to the *same* parameter, used transposed in
the second spot -- a "tied autoencoder" pattern. The two layers are
already exact shape-transposes of each other ((d_model, seg_len) vs.
(seg_len, d_model)), so this is a structural capacity reduction, not a
hyperparameter change: it removes seg_len*d_model parameters (the
predict head's independent weight) while leaving every activation shape
in the model unchanged.

Why this direction, not the opposite one: every Stage 2 strand that added
capacity to this architecture has regressed or landed neutral at best
(docs/stage2_new_strands_draft.md, docs/stage2_architecture_variants_draft.md);
the two genuine wins so far (d_model=256, Huber loss) both reduce
something rather than add it. This is the same direction applied
structurally -- enforce parameter sharing rather than just picking a
smaller d_model -- and untested elsewhere in this project.

Change (only): value embedding and predict share one weight matrix
(embed_weight, shape d_model x seg_len); predict biases remain
independent (bias isn't naturally tied by the transpose relationship).
Everything else (segment partition, GRU encoding, PMF decode, positional
embedding, normalization) is identical to models/SegRNN.py.

Only --dec_way pmf is supported (like SegRNNTime/SegRNNAttn/SegRNNRocket/SegRNNFFT).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        assert self.dec_way == 'pmf', "SegRNNWeightTied only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # tied weight -- reuse a plain nn.Linear's default init for the shared
        # matrix, then discard the Linear module itself (its .weight/.bias
        # Parameters are what get registered as this model's own attributes)
        _init_linear = nn.Linear(self.seg_len, self.d_model)
        self.embed_weight = _init_linear.weight  # (d_model, seg_len)
        self.embed_bias = _init_linear.bias      # (d_model,)
        self.predict_bias = nn.Parameter(torch.zeros(self.seg_len))
        self.predict_dropout = nn.Dropout(self.dropout)

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
        x_seg = x.reshape(-1, self.seg_num_x, self.seg_len)
        x = F.relu(F.linear(x_seg, self.embed_weight, self.embed_bias))

        # encoding
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

        # predict head reuses embed_weight, transposed -- the only structural change
        hy = self.predict_dropout(hy)
        y = F.linear(hy, self.embed_weight.t(), self.predict_bias)
        y = y.view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
