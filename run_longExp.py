import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# SegRNNTime (Stage 2 improvement): width of the per-timestep calendar feature
# vector from utils/timefeatures.py -- 4 for hourly data (ETTh1/ETTh2/Electricity),
# unused by every other model.
parser.add_argument('--mark_dim', type=int, default=4, help='SegRNNTime: calendar feature width')
parser.add_argument('--hour_emb_dim', type=int, default=16, help='SegRNNTime: hour-of-day embedding width')
parser.add_argument('--weekday_emb_dim', type=int, default=8, help='SegRNNTime: day-of-week embedding width')

# Stage 2 improvement (Dataset_ETT_hour only): swap the plain StandardScaler
# for a power transform, fit on train only. 0 = off (reconstruction
# unaffected), 1 = Yeo-Johnson (MLE-fit lambda), 2 = Box-Cox (MLE-fit
# lambda; dropped as a Stage 2 result, ETTh1 is not strictly positive),
# 3 = Box-Cox with a fixed lambda (--boxcox_lambda, also dropped),
# 4 = Yeo-Johnson with a fixed lambda (--yj_lambda), not fit via maximum
# likelihood -- Yeo-Johnson (unlike Box-Cox) is defined for all real
# values, so this needs no positivity workaround.
parser.add_argument('--power_transform', type=int, default=0,
                     help='0=off, 1=Yeo-Johnson (MLE), 2=Box-Cox (MLE), 3=Box-Cox (fixed --boxcox_lambda), 4=Yeo-Johnson (fixed --yj_lambda)')
parser.add_argument('--boxcox_lambda', type=float, default=0.0,
                     help='Fixed Box-Cox lambda, only used when --power_transform 3')
parser.add_argument('--yj_lambda', type=float, default=0.0,
                     help='Fixed Yeo-Johnson lambda, only used when --power_transform 4')

# SegRNNRocket (Stage 2 improvement): fixed random-convolution feature bank
parser.add_argument('--rocket_kernels', type=int, default=32, help='SegRNNRocket: number of random conv kernels')
parser.add_argument('--rocket_kernel_size', type=int, default=9, help='SegRNNRocket: kernel width')

# SegRNNFFT (Stage 2 improvement): top-K FFT-magnitude feature bank
parser.add_argument('--fft_k', type=int, default=32, help='SegRNNFFT: number of top FFT magnitude bins')

# SegRNNDeepEncoder (Stage 2 improvement): stacked GRU encoder
parser.add_argument('--encoder_layers', type=int, default=2, help='SegRNNDeepEncoder: number of stacked GRU layers')

# SegRNNConvEmbed (Stage 2 improvement): conv-based per-segment embedding
parser.add_argument('--conv_kernel_size', type=int, default=5, help='SegRNNConvEmbed: within-segment conv kernel width')

# SegRNNPoolContext (Stage 2 improvement): parameter-free pooling context
parser.add_argument('--pool_type', type=str, default='mean', help='SegRNNPoolContext: mean or max')

# SegRNNReservoir (Stage 2 improvement): frozen ESN-style reservoir cell
parser.add_argument('--reservoir_spectral_radius', type=float, default=0.9,
                     help='SegRNNReservoir: target spectral radius of the frozen recurrent weight matrix')
parser.add_argument('--reservoir_scale_init', type=int, default=1,
                     help='SegRNNReservoir: 1=scale to reservoir_spectral_radius before freezing, 0=freeze raw default init (ablation); True 1 False 0')

# Ensembling (Stage 2 improvement): save raw pred.npy/true.npy for this run
# so predictions can be averaged across seeds after the fact. 0 = off (no
# extra disk I/O), matches every other run unless explicitly requested.
parser.add_argument('--save_preds', type=int, default=0, help='Save results/<setting>/pred.npy and true.npy; True 1 False 0')

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function: mse, mae, huber, blend')
parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber loss delta')
parser.add_argument('--blend_alpha', type=float, default=0.5, help='blend loss: alpha*MSE + (1-alpha)*MAE')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.dropout,
            args.rnn_type,
            args.dec_way,
            args.seg_len,
            args.loss,
            args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.dropout,
        args.rnn_type,
        args.dec_way,
        args.seg_len,
        args.loss,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
