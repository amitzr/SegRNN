'''
Stage 2 improvement candidate: turns SegRNN's own recurrent cell into a
reservoir-computing / Echo State Network (ESN, Jaeger 2001) core, rather
than adding a reservoir-derived feature source alongside a trained
encoder (the way models/SegRNNRocket.py and models/SegRNNFFT.py did).
Neither reservoir computing nor ESNs appear anywhere in
docs/SegRNN_paper.pdf or this repo -- genuinely untested territory here.

Why this fits the project's own established pattern better than another
feature-injection strand: every strand that injected a *pooled* random
feature vector into h_n (ROCKET's Max/PPV, FFT's top-K magnitude) threw
away temporal order and recency before injecting it, and both regressed
(docs/stage2_new_strands_draft.md, docs/stage2_architecture_variants_draft.md).
An ESN reservoir's state is the *live* end-state of a random recurrence
unrolled over the actual sequence -- recency and order are preserved
structurally, not reconstructed after pooling. Separately, this project's
strongest results (d_model=256, Huber loss, pool context, weight tying)
all *reduce trained capacity* rather than add it. Freezing SegRNN's own
recurrent cell is a more radical version of that same lever: instead of
training a smaller GRU, don't train the recurrent core at all -- only
the value embedding, positional embeddings, and predict head (SegRNN's
own analogue of an ESN's linear "readout") are trained. That's roughly
5-10% of the reconstruction's trainable parameter count.

Two things that make this a *properly built* ESN rather than "a frozen
GRU dressed up in ESN language":
1. rnn_type must be 'rnn' (a plain tanh cell), not 'gru' or 'lstm' --
   GRU/LSTM gates are meant to be *trained* to control information flow;
   freezing them randomly is a much weaker, less-motivated experiment
   than freezing a cell whose recurrent formulation matches the
   classical ESN reservoir directly.
2. The recurrent weight matrix's spectral radius must be scaled to the
   standard ESN range (~0.9-0.99 by default here, --reservoir_spectral_radius)
   at initialization -- PyTorch's default RNN init is not tuned for the
   echo-state property (stable, fading memory), and an untuned random
   recurrent matrix can be chaotic (radius >> 1, unstable dynamics) or
   have almost no memory (radius << 1) before it's ever trained, which
   frozen weights can never fix later. --reservoir_scale_init 0 disables
   this (keeps PyTorch's raw default init instead) as a deliberate
   ablation: does *properly initialized* freezing behave differently
   from *naive* freezing?

Change (only): rnn_type is asserted to 'rnn'. After construction, the
recurrent weight matrix (weight_hh_l0) is optionally rescaled so its
largest eigenvalue's magnitude equals --reservoir_spectral_radius
(default 0.9), then every parameter of the cell has requires_grad set to
False. Encode and decode still share this one (now frozen) cell, exactly
as in models/SegRNN.py -- everything else (segment partition, value
embedding, PMF decode, positional embedding, normalization) is
unchanged.

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

        self.reservoir_spectral_radius = getattr(configs, 'reservoir_spectral_radius', 0.9)
        self.reservoir_scale_init = getattr(configs, 'reservoir_scale_init', 1)

        assert self.rnn_type == 'rnn', "SegRNNReservoir only supports --rnn_type rnn (see module docstring)"
        assert self.dec_way == 'pmf', "SegRNNReservoir only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- identical to models/SegRNN.py except the cell gets frozen below
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)

        if self.reservoir_scale_init:
            self._scale_spectral_radius()
        for p in self.rnn.parameters():
            p.requires_grad = False

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

    def _scale_spectral_radius(self):
        """Rescale weight_hh_l0 so its largest-magnitude eigenvalue equals
        reservoir_spectral_radius -- standard ESN reservoir initialization
        (echo-state property), done once, before freezing."""
        with torch.no_grad():
            w = self.rnn.weight_hh_l0
            eigvals = torch.linalg.eigvals(w)
            current_radius = eigvals.abs().max().item()
            if current_radius > 1e-8:
                w.mul_(self.reservoir_spectral_radius / current_radius)

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

        # encoding -- through the frozen reservoir cell
        _, hn = self.rnn(x)  # hn: 1,bc,d

        # decoding (pmf only, see the dec_way assert above) -- same frozen cell
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
