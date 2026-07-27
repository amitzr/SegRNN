'''
Stage 2 improvement candidate: isolates one variable that
models/SegRNNBidir.py changed two of at once. SegRNNBidir made the
encoder bidirectional *and*, as a forced consequence, gave the decoder
its own separate (unshared) GRU cell -- a bidirectional encoder can't be
reused for a single-step decode, so weight-sharing broke automatically.
That strand came back "around the same" as the reconstruction, a genuine
surprise (docs/stage2_architecture_variants_draft.md predicted it would
be more likely to regress than encoder depth, since it adds the most
capacity of any strand tested -- instead depth regressed and this one
didn't).

This model keeps the encoder unidirectional (identical to models/SegRNN.py)
and *only* un-shares the decode cell, giving it its own independently-
learned GRU with the same shape as the encoder's. If this also comes back
neutral, that points at un-sharing (not bidirectionality) as whatever is
absorbing SegRNNBidir's extra capacity without hurting -- a genuinely new
finding about SegRNN's efficiency-driven weight-sharing design. If this
regresses (unlike SegRNNBidir), that would instead suggest bidirectional
context specifically was pulling its weight and compensating for the
unshared decoder's added capacity.

Change (only): a second nn.GRU (decode_rnn), same shape as the encoder's,
independently initialized -- no weight sharing between encode and decode.
Everything else (segment partition, value embedding, positional
embedding, predict head, normalization) is identical to models/SegRNN.py.

Only --dec_way pmf is supported, and only rnn_type='gru' (matching
SegRNNBidir, for a clean comparison between the two).
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

        assert self.rnn_type == 'gru', "SegRNNUnshared only supports --rnn_type gru (see module docstring)"
        assert self.dec_way == 'pmf', "SegRNNUnshared only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- identical to models/SegRNN.py
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.encoder_rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                                   batch_first=True, bidirectional=False)

        # decoder -- separate, independently-learned cell (the only structural change)
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

        # encoding (unidirectional, same as models/SegRNN.py)
        _, hn = self.encoder_rnn(x)  # hn: 1,bc,d

        # decoding (pmf only, see the dec_way assert above) -- separate cell
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        _, hy = self.decode_rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
