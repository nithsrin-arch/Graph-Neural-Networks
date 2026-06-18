from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#This is the main function that will be used to launch the training and testing of the GNN
def main():
    parser = argparse.ArgumentParser(description='Quantum Circuit GNN project launcher')
    sub = parser.add_subparsers(dest='mode', required=True)
# This is the parser for the training of the GNN
    p_train = sub.add_parser('train', help='Train the GNN')
    p_train.add_argument('--train_data', required=True)
    p_train.add_argument('--val_data', default=None)
    p_train.add_argument('--val_ratio', type=float, default=0.1)
    p_train.add_argument('--out_dir', default='runs/quantum_gnn')
    p_train.add_argument('--epochs', type=int, default=200)
    p_train.add_argument('--batch_size', type=int, default=64)
    p_train.add_argument('--lr', type=float, default=0.01)
    p_train.add_argument('--weight_decay', type=float, default=1e-4)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--acc_tol', type=float, default=0.1)
    p_train.add_argument('--device', default=None)


# This is the parser foer the testing of the GNN
    p_test = sub.add_parser('test', help='Test the trained GNN')
    p_test.add_argument('--test_data', required=True)
    p_test.add_argument('--checkpoint', required=True)
    p_test.add_argument('--out_dir', default='runs/quantum_gnn_test')
    p_test.add_argument('--acc_tol', type=float, default=0.1)
    p_test.add_argument('--device', default=None)

    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    if args.mode == 'train':
        cmd = [
            sys.executable, str(root / 'train_gnn.py'),
            '--train_data', args.train_data,
            '--out_dir', args.out_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--seed', str(args.seed),
            '--val_ratio', str(args.val_ratio),
            '--acc_tol', str(args.acc_tol),
        ]
        if args.val_data:
            cmd.extend(['--val_data', args.val_data])
        if args.device:
            cmd.extend(['--device', args.device])
        subprocess.run(cmd, check=True)
    else:
        cmd = [
            sys.executable, str(root / 'test_gnn.py'),
            '--test_data', args.test_data,
            '--checkpoint', args.checkpoint,
            '--out_dir', args.out_dir,
            '--acc_tol', str(args.acc_tol),
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
