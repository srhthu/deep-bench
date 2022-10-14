import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('op', help = 'operator: sum or mul')
    parser.add_argument('--n', type = int, default = 10000, help = 'num of operations (K)')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    x_a = torch.rand(1000, 1000).cuda()
    x_b = torch.rand(1000, 1000).cuda() if args.op == 'sum' else torch.rand(1000, 64).cuda()

    start = time.time()
    for _ in tqdm(range(args.n * 1000)):
        if args.op == 'sum':
            _ = x_a + x_b
        elif args.op == 'mul':
            _ = torch.matmul(x_a, x_b)
    end = time.time()
    dur = end - start
    print(f'{args.op} {args.n}K : {dur:.4f}s')

if __name__ == '__main__':
    main()