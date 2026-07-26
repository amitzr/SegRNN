'''
Stage 2 improvement over models/SegRNN.py: wires the calendar/time
features (hour-of-day, day-of-week, day-of-month, day-of-year -- computed
by data_provider/data_loader.py via utils/timefeatures.py, but discarded
on the original SegRNN path, see docs/data_pipeline_audit.md section 7)
into the encoder.

Change (only): each input segment's raw values (shape seg_len) are
concatenated with (a) an embedding of that segment's hour-of-day
(nn.Embedding(24, hour_emb_dim), looked up from the segment's last
timestep) and (b) its remaining calendar features (day-of-week,
day-of-month, day-of-year, also last-timestep, kept as raw scalars) --
before the Linear(w+hour_emb_dim+3 -> d)+ReLU projection. Motivation:
ETTh1 has a strong daily/weekly cycle (confirmed by the seasonal-naive
baseline in docs/stage1_report_draft.md beating plain naive by a wide
margin); the original architecture has no explicit signal telling the
encoder which part of that cycle a given input segment belongs to, and
must infer it purely from the recurrent state.

Design grounded in docs/DL_for_TS.pdf (course lecture), "Include the year
and month as predictor variables" / "Improved Model" / "Embedding Layer":
the taught pattern for injecting a calendar feature is to route the
cyclical/categorical part (there: month, 12 categories) through a learned
Embedding, and only treat genuinely linear parts (there: year) as raw
scalars -- not to concatenate every calendar feature as a raw number.
Applied here: hour-of-day (24 categories, the dominant daily-cycle driver
per the seasonal-naive baseline) gets the Embedding; day-of-week/month/
year stay raw, same as the previous attempt.

Two earlier, less-grounded attempts are recorded in
docs/stage2_report_draft.md as negative results, both concatenating
HourOfDay as a single raw continuous scalar (in [-0.5, 0.5]) instead of
embedding it -- giving the network very little room to represent 24
distinct, non-linearly-related hour identities from one number:
1. Mean-pooling each segment's calendar features. seg_len=24 for hourly
   ETTh1 means each segment spans exactly one full day, so mean-pooling
   HourOfDay across a full 24h cycle washes it out to nearly the same
   constant for every segment, destroying the signal. Measurably *hurt*
   MSE/MAE vs. the plain reconstruction.
2. Last-timestep raw scalar (no pooling issue, but still just one
   continuous number for a 24-way categorical quantity). Also
   consistently *hurt* MSE/MAE, worsening with horizon.

Everything else (GRU encoding, PMF decoding, normalization) is unchanged
from models/SegRNN.py -- see docs/architecture_notes.md for the full
block-by-block mapping, which still applies here except for the segment
embedding step described above.
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
        self.mark_dim = configs.mark_dim  # width of the per-timestep calendar feature vector
        self.hour_emb_dim = configs.hour_emb_dim

        # Assumes utils.timefeatures's hourly-frequency column order:
        # [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] -- column 0 is the
        # one routed through the embedding below. Hourly-ETT-specific by
        # design (see module docstring); a different --freq would need a
        # different column index here.
        assert self.mark_dim >= 1, "SegRNNTime needs at least an HourOfDay column in mark_dim"

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        # rmf is not supported here: its recursive step re-embeds a predicted
        # segment (no calendar features available for it) through the same
        # valueEmbedding layer, whose input width now includes mark_dim --
        # shapes wouldn't match. pmf (this repo's reference config) has no
        # such issue since it embeds only real input segments.
        assert self.dec_way == 'pmf', "SegRNNTime only supports --dec_way pmf"

        self.seg_num_x = self.seq_len//self.seg_len

        # hour-of-day (24 categories) gets a learned embedding, per
        # docs/DL_for_TS.pdf's "embed the cyclical/categorical feature"
        # pattern; the remaining mark_dim-1 calendar features stay raw.
        self.hour_embedding = nn.Embedding(24, self.hour_emb_dim)

        # build model -- only this Linear's input width differs from SegRNN.py
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len + self.hour_emb_dim + (self.mark_dim - 1), self.d_model),
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

        self.seg_num_y = self.pred_len // self.seg_len

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

    def forward(self, x, x_mark):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y k:mark_dim
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment values    b,c,s -> bc,n,w
        x_seg = x.reshape(-1, self.seg_num_x, self.seg_len)

        # segment calendar features: take each segment's last timestep (not a
        # mean -- see the module docstring for why pooling backfires here),
        # b,s,k -> b,n,w,k -> last of w -> b,n,k
        mark_seg = x_mark.reshape(batch_size, self.seg_num_x, self.seg_len, self.mark_dim)[:, :, -1, :]

        # HourOfDay (column 0) is continuous in [-0.5, 0.5] (utils/timefeatures.py's
        # HourOfDay = hour/23.0 - 0.5); invert it back to an integer 0-23
        # bucket and look up its embedding, per the module docstring.
        hour_idx = torch.clamp(torch.round((mark_seg[..., 0] + 0.5) * 23), 0, 23).long()  # b,n
        hour_emb = self.hour_embedding(hour_idx)  # b,n,hour_emb_dim
        mark_feat = torch.cat([hour_emb, mark_seg[..., 1:]], dim=-1)  # b,n,hour_emb_dim+mark_dim-1

        # repeat per channel -> bc,n,hour_emb_dim+mark_dim-1
        mark_feat = mark_feat.unsqueeze(1).repeat(1, self.enc_in, 1, 1).reshape(-1, self.seg_num_x, mark_feat.size(-1))

        # embedding    bc,n,w+hour_emb_dim+mark_dim-1 -> bc,n,d
        x_emb = self.valueEmbedding(torch.cat([x_seg, mark_feat], dim=-1))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x_emb)
        else:
            _, hn = self.rnn(x_emb)  # bc,n,d  1,bc,d

        # decoding (pmf only, see the dec_way assert above) -- identical to models/SegRNN.py
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
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
