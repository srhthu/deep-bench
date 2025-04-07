import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import platform
import re
import pandas as pd
from datetime import datetime

from transformers.models.bert.modeling_bert import BertForSequenceClassification

def get_cpu_brand():
    with open('/proc/cpuinfo') as f:
        lines = [k.strip() for k in f]
    cpu_brand = []
    for k in lines:
        m = re.match(r"model name[\s]*:[\s]*(.*)$", k)
        if m is not None:
            cpu_brand.append(m.group(1))
    return list(set(cpu_brand))[0]

def get_gpu_brands():
    stream = os.popen('nvidia-smi --query-gpu=index,name --format=csv')
    lines = stream.read().split('\n')
    gpus = [k.split(',')[1].strip() for k in lines[1:] if len(k) > 0]
    return gpus

def get_gpu_brand():
    gid = int(os.environ['CUDA_VISIBLE_DEVICES'])
    return get_gpu_brands()[gid]

def get_df():
    df = pd.DataFrame(columns = [
        'date', 'hostname', 'gpu', 'cpu', 'nbatch', 
        'bs', 'duration', 'speed_it', 'speed_sample'])
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action = 'store_true', help = 'write to results.csv')
    parser.add_argument('-n', type = int, default = 100, help = 'number of batches')
    parser.add_argument('--cpu', action = 'store_true')
    parser.add_argument('--bs', type = int, default = 6, help = 'batch size')
    parser.add_argument('--infer', action = 'store_true', help = 'only do inference')

    args = parser.parse_args()

    return args

def train_model(model, optim, token_ids, labels):
    print(f'Total iteration: {len(token_ids)}')
    start = time.time()
    model.train()
    n = len(token_ids)
    for i in tqdm(range(n)):
        x = token_ids[i]
        y = labels[i]
        optim.zero_grad()
        outputs = model(x, labels = y)
        outputs.loss.backward()
       # optim.step()
    end = time.time()
    duration = end - start
    return duration

def main():
    args = get_args()

    token_ids = torch.randint(0, 1000, (args.n, args.bs, 512))
    labels = torch.randint(0,2, (args.n, args.bs))

    print('Build model...')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)

    if not args.cpu:
        print('Migrate data to gpu')
        model.cuda()
        token_ids = token_ids.cuda()
        labels = labels.cuda()

    print('Build optimizer')
    optim = torch.optim.AdamW(model.parameters(), lr = 1e-5)

    duration = train_model(model, optim, token_ids, labels)
    speed_iter = args.n / duration
    speed_sample = args.n * args.bs / duration
    #print(f'Time: {duration:.2f}s {speed_iter:.2f}it/s  {speed_sample:.2f}sample/s')

    df = get_df()
    df = df.append({
        "date": datetime.now(),
        "hostname": platform.node(),
        "gpu": get_gpu_brand(),
        "cpu": get_cpu_brand(),
        "nbatch": args.n,
        "bs": args.bs,
        "duration": duration,
        "speed_it": speed_iter,
        "speed_sample": speed_sample
    }, ignore_index = True)

    df['date'] = df['date'].dt.floor('s')
    for col in ['duration', 'speed_it', 'speed_sample']:
        df[col] = df[col].round(4)
    if args.write:
        header = not os.path.exists('results.csv')
        df.to_csv('results.csv', header = header, index = False, mode = 'a')
    else:
        print(df.to_csv(index = False))

if __name__ == '__main__':
    main()
