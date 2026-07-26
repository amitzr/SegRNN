'''
Stage 2 improvement over models/SegRNN.py, attempt 4. Three earlier
attempts (recorded in docs/stage2_report_draft.md) all injected calendar
features into the *encoder* -- concatenated with each input segment's raw
values, first mean-pooled, then last-timestep-raw, then last-timestep
through an hour-of-day embedding. All three consistently *hurt* MSE/MAE
vs. the plain reconstruction, worsening with horizon.

Diagnosis (course lecture, docs/DL_for_TS.pdf, "Attention" slide): a
single encoder context vector is an "information bottleneck" -- SegRNN's
encoder already compresses the whole look-back window into one hidden
state h_n before the decoder ever sees it. Every encoder-side calendar
addition forced calendar information to compete with raw-value
information for space in that one vector, which plausibly explains why
representation quality (mean/last-timestep/embedding) barely mattered --
the bottleneck was the problem, not the representation.

This attempt moves the calendar signal to the *decoder* instead, where it
doesn't compete for encoder capacity at all. SegRNN's PMF decoder already
has a side-channel built for exactly this purpose: the positional
embedding PE = concat(rp, cp) (relative-position + channel-position),
fed directly alongside the repeated h_n into the decode step, bypassing
the encoder bottleneck entirely. The paper's own ablation (Table V) shows
PE -- specifically RP -- is the single highest-leverage component in the
whole architecture (28.8% MSE reduction alone), so this targets the part
of the model already demonstrated to matter most.

Change: PE is extended to concat(rp, cp, hour_embedding, weekday_embedding),
where hour_embedding/weekday_embedding are learned nn.Embedding lookups
(24 and 7 categories respectively, per docs/DL_for_TS.pdf's "embed the
cyclical/categorical feature, keep linear ones raw" pattern) of each
target segment's own future hour-of-day and day-of-week -- known in
advance for the whole forecast horizon (calendar facts, not data), so
this is not leakage. The encoder goes back to being identical to
models/SegRNN.py -- no calendar features there at all, isolating what's
actually being tested (decoder-side injection) from the three earlier,
now-abandoned encoder-side attempts.

Everything else (GRU encoding, RMF -> not supported, same as before;
normalization) is unchanged from models/SegRNN.py -- see
docs/architecture_notes.md for the full block-by-block mapping.
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
        self.mark_dim = configs.mark_dim
        self.hour_emb_dim = configs.hour_emb_dim
        self.weekday_emb_dim = configs.weekday_emb_dim

        # Assumes utils.timefeatures's hourly-frequency column order:
        # [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] -- columns 0 and 1
        # are the ones routed through embeddings below. Hourly-ETT-specific
        # by design; a different --freq would need different indices.
        assert self.mark_dim >= 2, "SegRNNTime needs HourOfDay and DayOfWeek columns in mark_dim"

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNTime only supports --dec_way pmf (see module docstring)"

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # encoder -- identical to models/SegRNN.py, no calendar features here
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

        # decoder PE: rp (+cp) as in SegRNN.py, extended with hour/weekday
        # embeddings of the target segment's own future calendar values.
        self.hour_embedding = nn.Embedding(24, self.hour_emb_dim)
        self.weekday_embedding = nn.Embedding(7, self.weekday_emb_dim)

        pe_static_dim = self.d_model - self.hour_emb_dim - self.weekday_emb_dim
        assert pe_static_dim > 0, "d_model too small for hour_emb_dim + weekday_emb_dim"
        if self.channel_id:
            assert pe_static_dim % 2 == 0, \
                "d_model - hour_emb_dim - weekday_emb_dim must be even when channel_id=1 (split between rp/cp)"
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, pe_static_dim // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, pe_static_dim // 2))
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, pe_static_dim))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)

    def forward(self, x, y_mark):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y k:mark_dim
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d (unchanged from SegRNN.py)
        x_emb = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x_emb)
        else:
            _, hn = self.rnn(x_emb)  # bc,n,d  1,bc,d

        # future calendar embeddings for each of the m target segments --
        # known in advance, not leakage. y_mark's last pred_len rows are the
        # forecast horizon (matches how dec_inp is sliced in exp_main.py).
        y_mark = y_mark[:, -self.pred_len:, :]
        y_mark_seg = y_mark.reshape(batch_size, self.seg_num_y, self.seg_len, self.mark_dim)[:, :, -1, :]  # b,m,k

        # HourOfDay/DayOfWeek are continuous in [-0.5, 0.5] (utils/timefeatures.py);
        # invert back to integer buckets and look up their embeddings.
        hour_idx = torch.clamp(torch.round((y_mark_seg[..., 0] + 0.5) * 23), 0, 23).long()  # b,m
        weekday_idx = torch.clamp(torch.round((y_mark_seg[..., 1] + 0.5) * 6), 0, 6).long()  # b,m
        cal_emb = torch.cat([self.hour_embedding(hour_idx), self.weekday_embedding(weekday_idx)], dim=-1)  # b,m,H+W
        cal_emb = cal_emb.unsqueeze(1).repeat(1, self.enc_in, 1, 1)  # b,c,m,H+W (same future dates for every channel)

        # decoding: PE = concat(rp, cp, cal_emb), replacing SegRNN.py's PE = concat(rp, cp)
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1)  # c,m,pe_static_dim
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # b,c,m,pe_static_dim
        else:
            pos_emb = self.pos_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.enc_in, 1, 1)  # b,c,m,pe_static_dim

        pe = torch.cat([pos_emb, cal_emb], dim=-1).reshape(-1, 1, self.d_model)  # bcm,1,d_model

        if self.rnn_type == "lstm":
            _, (hy, cy) = self.rnn(pe,
                                   (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                    cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
        else:
            _, hy = self.rnn(pe, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
