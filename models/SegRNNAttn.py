'''
Stage 2 improvement candidate, independent of the calendar-feature work
in models/SegRNNTime.py: replaces the encoder-decoder "information
bottleneck" itself, rather than adding more information for it to carry.

Diagnosis (docs/DL_for_TS.pdf's Attention slide, explicit
"Information Bottleneck" label): a single encoder context vector forces
everything the encoder has seen into one fixed-size representation before
the decoder ever sees it. models/SegRNN.py's h_n -- the final GRU hidden
state after all n=seq_len/seg_len segments -- is exactly this: the
decoder only ever sees h_n (repeated), never any intermediate segment
state. docs/stage2_report_draft.md's calendar-feature attempts found
that adding information competes for space in this one vector, which is
one plausible explanation for why three encoder-side attempts all
regressed. This model tests the complementary fix: leave h_n's *content*
alone, and instead let the decoder attend over *all* n encoder segment
states (not just the last one) at each decode step.

Change (only): the encoder now also returns its full per-segment output
sequence enc_outputs (b*c, n, d), not just h_n. In the PMF decode step,
each of the m target positions' positional embedding pe_j is used as an
attention query over enc_outputs (keys/values via learned projections),
producing a per-position context vector; pe_j + context_j (not pe_j
alone) is what's fed into the shared decode GRU call alongside the
repeated h_n. Standard scaled dot-product attention
(docs/DL_for_TS.pdf's Attention/Transformer slides).

Only --dec_way pmf is supported (like SegRNNTime): rmf's recursive decode
would need per-step attention recomputation, a different design not
attempted here. Everything else (segment partition, value embedding, GRU
encoding, normalization) is unchanged from models/SegRNN.py.
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

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way == 'pmf', "SegRNNAttn only supports --dec_way pmf (see module docstring)"

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

        # attention over encoder segment states -- the only structural addition
        self.attn_query = nn.Linear(self.d_model, self.d_model)
        self.attn_key = nn.Linear(self.d_model, self.d_model)
        self.attn_value = nn.Linear(self.d_model, self.d_model)
        self.attn_scale = self.d_model ** 0.5

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

        # decoding (pmf only, see the dec_way assert above)
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)  # bcm,1,d
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)  # bcm,1,d

        # attention: each of the m target positions attends over this (b,c)
        # pair's n encoder segment states. enc_outputs (bc,n,d) is expanded
        # to (bcm,n,d) -- m consecutive copies per (b,c) block, matching
        # pos_emb's own (b,c,m) flattening order below.
        keys = self.attn_key(enc_outputs)      # bc,n,d
        values = self.attn_value(enc_outputs)  # bc,n,d
        keys = keys.unsqueeze(1).repeat(1, self.seg_num_y, 1, 1).reshape(-1, self.seg_num_x, self.d_model)
        values = values.unsqueeze(1).repeat(1, self.seg_num_y, 1, 1).reshape(-1, self.seg_num_x, self.d_model)

        query = self.attn_query(pos_emb)  # bcm,1,d
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.attn_scale  # bcm,1,n
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values)  # bcm,1,d

        decoder_input = pos_emb + context  # bcm,1,d

        if self.rnn_type == "lstm":
            _, (hy, cy) = self.rnn(decoder_input,
                                   (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                    cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
        else:
            _, hy = self.rnn(decoder_input, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y
