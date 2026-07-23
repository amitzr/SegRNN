"""
Classical baselines (naive, seasonal naive) evaluated on the exact same
data pipeline as run_longExp.py: same Dataset classes, same chronological
split, same StandardScaler (fit on train only), same scaled-space metric
convention (this repo's exp_main.py reports MSE/MAE on scaler-transformed
values, not inverse-transformed originals -- see docs/data_pipeline_audit.md
section 5). This makes the numbers printed here directly comparable to
run_longExp.py's own console output for the same --data/--seq_len/--pred_len.

Usage (mirrors run_longExp.py's relevant flags):
    python scripts/baselines.py --data ETTh1 --root_path ./dataset/ \
        --data_path ETTh1.csv --seq_len 720 --pred_len 96

No training, no GPU, no randomness -- naive and seasonal-naive forecasts
are deterministic functions of the input window.
"""
import argparse
import csv
import datetime
import math
import os
import sys

# allow `python scripts/baselines.py` to find the repo-root packages
# (data_provider, utils) regardless of the current working directory --
# unlike run_longExp.py, which lives at the repo root itself, this script
# is one directory down, so Python's default sys.path[0] (this file's own
# directory) doesn't include them.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from utils.metrics import MAE, MSE

DATA_DICT = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

# period (in rows) of one full seasonal cycle, per the assignment's spec:
# 24 for hourly ETT/Electricity, 96 for 15-min ETTm, 144 for 10-min Weather.
DEFAULT_PERIOD_BY_DATA = {
    'ETTh1': 24,
    'ETTh2': 24,
    'ETTm1': 96,
    'ETTm2': 96,
}
DEFAULT_PERIOD_BY_FILENAME = {
    'electricity': 24,
    'weather': 144,
}

RUNS_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'runs.csv')
RUNS_CSV_HEADER = [
    'run_id', 'timestamp', 'model', 'dataset', 'horizon', 'seq_len', 'seg_len',
    'd_model', 'seed', 'flags', 'mse', 'mae', 'mase', 'epoch_time_s', 'params',
    'peak_mem_mb', 'notes',
]


def infer_period(args):
    if args.period is not None:
        return args.period
    if args.data in DEFAULT_PERIOD_BY_DATA:
        return DEFAULT_PERIOD_BY_DATA[args.data]
    stem = os.path.splitext(os.path.basename(args.data_path))[0].lower()
    for key, period in DEFAULT_PERIOD_BY_FILENAME.items():
        if key in stem:
            return period
    raise ValueError(
        f"Can't infer seasonal period for --data {args.data} --data_path "
        f"{args.data_path}; pass --period explicitly."
    )


def build_dataset(args, flag):
    Data = DATA_DICT[args.data]
    return Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        # timeenc=1 (not 0): this script never reads data_stamp -- it only
        # needs data_x -- and timeenc=0's code path in data_provider/data_loader.py
        # calls df_stamp.drop(['date'], 1) with a positional axis arg, which
        # pandas 2.x removed (TypeError). timeenc=1 takes a different branch
        # that avoids it. data_provider/ is upstream code this session
        # doesn't modify, so route around the bug here instead of patching it.
        timeenc=1,
        freq=args.freq,
    )


def windowed_forecasts(data_x, seq_len, pred_len, period):
    """Build (trues, naive_preds, seasonal_preds), each shape (n_windows, pred_len, n_channels).

    Mirrors Dataset.__getitem__'s windowing exactly: window i covers
    input [i : i+seq_len), target [i+seq_len : i+seq_len+pred_len).
    """
    n_windows = len(data_x) - seq_len - pred_len + 1
    trues, naive_preds, seasonal_preds = [], [], []
    reps = math.ceil(pred_len / period)
    for i in range(n_windows):
        s_end = i + seq_len
        true = data_x[s_end:s_end + pred_len]
        last_value = data_x[s_end - 1]
        naive_pred = np.tile(last_value, (pred_len, 1))
        last_cycle = data_x[s_end - period:s_end]
        seasonal_pred = np.tile(last_cycle, (reps, 1))[:pred_len]

        trues.append(true)
        naive_preds.append(naive_pred)
        seasonal_preds.append(seasonal_pred)

    return np.stack(trues), np.stack(naive_preds), np.stack(seasonal_preds)


def seasonal_naive_scale(train_x, period):
    """In-sample seasonal-naive MAE per channel, computed on the training split only (MASE denominator)."""
    errors = np.abs(train_x[period:] - train_x[:-period])
    return errors.mean(axis=0)  # shape (n_channels,)


def mase(preds, trues, scale):
    """Mean absolute scaled error: per-channel MAE / per-channel in-sample seasonal-naive MAE, averaged over channels."""
    per_channel_mae = np.abs(preds - trues).mean(axis=(0, 1))  # shape (n_channels,)
    return float(np.mean(per_channel_mae / scale))


def append_run(row):
    write_header = not os.path.exists(RUNS_CSV)
    with open(RUNS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RUNS_CSV_HEADER)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Naive / seasonal-naive baselines on the SegRNN data pipeline')
    parser.add_argument('--data', type=str, required=True, help='dataset type, e.g. ETTh1')
    parser.add_argument('--root_path', type=str, required=True, help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, help='data file name')
    parser.add_argument('--seq_len', type=int, required=True, help='input sequence length')
    parser.add_argument('--pred_len', type=int, required=True, help='prediction (horizon) length')
    parser.add_argument('--features', type=str, default='M', help='[M, S, MS], same meaning as run_longExp.py')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='data frequency, same meaning as run_longExp.py')
    parser.add_argument('--period', type=int, default=None,
                         help='seasonal period in rows; inferred from --data/--data_path if omitted')
    parser.add_argument('--seed', type=int, default=2024,
                         help='logged for consistency with run_longExp.py; these baselines are deterministic')
    args = parser.parse_args()

    np.random.seed(args.seed)
    period = infer_period(args)

    train_ds = build_dataset(args, 'train')
    test_ds = build_dataset(args, 'test')

    scale = seasonal_naive_scale(train_ds.data_x, period)
    trues, naive_preds, seasonal_preds = windowed_forecasts(
        test_ds.data_x, args.seq_len, args.pred_len, period
    )

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    results = {}
    for name, preds in [('naive', naive_preds), ('seasonal_naive', seasonal_preds)]:
        mse_val = MSE(preds, trues)
        mae_val = MAE(preds, trues)
        mase_val = mase(preds, trues, scale)
        results[name] = (mse_val, mae_val, mase_val)
        print(f'{name}: mse={mse_val:.7f}, mae={mae_val:.7f}, mase={mase_val:.7f}')

        run_id = f'{name}_{args.data}_{args.pred_len}_{timestamp}'
        flags = f'period={period};features={args.features};seq_len={args.seq_len}'
        append_run([
            run_id, timestamp, name, args.data, args.pred_len, args.seq_len,
            '', '', args.seed, flags, mse_val, mae_val, mase_val, '', '', '',
            'deterministic baseline, no training',
        ])


if __name__ == '__main__':
    main()
