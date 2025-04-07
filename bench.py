"""
Benchmark the GPU Performance of Transformer model.
"""

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import json
import platform
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

from pynvml import *
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType

class NVML_Mem:
    """Get GPU Memory Usage"""
    def __init__(self, gpu_indexes = None):
        if gpu_indexes is not None:
            self.gpu_indexes = gpu_indexes
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            self.gpu_indexes = [int(k) for k in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        else:
            self.gpu_indexes = list(range(torch.cuda.device_count()))
        nvmlInit()
    
    def __call__(self):
        return [self.get_mem_by_id(k) for k in self.gpu_indexes]
    
    def get_mem_by_id(self, index):
        h = nvmlDeviceGetHandleByIndex(index)
        info = nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024**3

def build_model_from_config(model_path):
    # load config from local file
    logging.info('load config')
    config = AutoConfig.from_pretrained(model_path)
    
    # initialize model
    logging.info('Initialize model.')
    model = AutoModelForCausalLM.from_config(config)
    return model

def build_model(
    model_path = './config/qwen2.5-instruct',
    load_weights = False,
    use_lora = False
):
    """
    Build model and place it on correct devices.
    """
    # build model with random initialization or loading weights
    if not load_weights:
        model = build_model_from_config(model_path)
    else:
        logging.info('load model')
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype = torch.bfloat16, device_map = {'':0}
        )
    
    # print number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total params: {total_params / 1000**3:.4f}B')

    # handle lora
    if use_lora:
        lora_config = LoraConfig(
            r=8,  # rank
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM  # or SEQ_CLS, TOKEN_CLS, etc.
        )
        model = get_peft_model(model, lora_config)
        tr_params, _ = model.get_nb_trainable_parameters()
        logging.info(f'Lora params: {tr_params / 1000**3:.4f}B || %: {100*tr_params/total_params:.2f}')
    
    # set dtype and move to GPU
    logging.debug('Move model to GPU')
    model.to(dtype = torch.bfloat16, device = Accelerator().local_process_index)
    logging.info(f'Device: {model.device}, dtype: {model.dtype}')

    return model

def train(
    model,
    max_step, 
    max_time,
    bs,
    seq_len,
    grad_acc_step = 1
):
    # initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps = grad_acc_step)

    # prepare optimizer and model
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    # prepare sudo input
    input_ids = torch.randint(1000, 10000, (bs, seq_len)).cuda(0)
    # run the first step
    _ = model(input_ids)

    start_time = time.time()
    logging.info('Start training ...')
    tbar = tqdm(
        range(max_step), 
        ncols = 80, 
        disable = accelerator.local_process_index > 0
    )
    for step in tbar:
        with accelerator.accumulate(model):
            loss = model(input_ids, labels = input_ids).loss
            loss = loss / grad_acc_step
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        # check stop for max_time
        cur_time = time.time()
        to_stop = torch.tensor([int(cur_time - start_time > max_time)], device = accelerator.device)
        gathered_stop = accelerator.gather(to_stop).tolist()
        if sum(gathered_stop) > 0:
            tbar.close()
            logging.info('stop. ' + repr(gathered_stop))
            break
    tbar.close()

    return {'step': step + 1, 'time': cur_time - start_time}

def calculate_speed(step, time, bs, seq_len, world_size):
    return bs * seq_len * world_size * step / time

def log_save_record(args, step, tr_time):
    """Get the test record, log and save it"""
    # copy the args
    record = {**args.__dict__}
    
    # add training records
    mem_usage_list = NVML_Mem()()
    speed = calculate_speed(
            step, tr_time, args.bs, args.seq_len, args.world_size
        )
    record.update({'step': step,
                   'tr_time': tr_time,
                   'token/s': speed,
                   'mem_usage': mem_usage_list})
    
    logging.info(repr(record))

    Path('./results').mkdir(exist_ok = True)
    ori_m = args.model.lower()
    mn = 'gpt2' if 'gpt2' in ori_m else 'qwen2' if 'qwen2' in ori_m else ''
    save_name = f'{mn}_{args.world_size}gpu_{time.strftime("%Y%m%d_%H%M")}'
    with open(f'./results/{save_name}', 'w') as f:
        json.dump(record, f, indent=  4)

def main():
    # prepare env
    accelerator = Accelerator()
    is_first = accelerator.local_process_index == 0
    world_size = accelerator.num_processes

    rank = accelerator.local_process_index
    # print(f'{rank} before: {NVML_Mem()()}')
    # accelerator.wait_for_everyone()
    # model = torch.nn.Linear(100000,10000).to(dtype = torch.bfloat16, device = rank)
    # accelerator.wait_for_everyone()
    # print(f'{rank} after: {NVML_Mem()()}')
    # model = accelerator.prepare(model)
    # print(f'{rank} prepare: {NVML_Mem()()} {list(model.parameters())[0].device}')
    # time.sleep(10)
    # exit()

    # Set logging format
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # disable logging for other processes
    if not is_first:
        logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                      help = 'model path, e.g., Qwen/Qwen2.5-7B-Instruct', 
                      default = './config/qwen2.5-instruct')
    parser.add_argument('--load', action = 'store_true')
    parser.add_argument('--lora', action = 'store_true')
    parser.add_argument('--max_step', type = int, default = 1024)
    parser.add_argument('--max_time', type = int, default = 20)
    parser.add_argument('--bs', type = int, default = 1)
    parser.add_argument('--seq_len', type = int, default = 512)
    parser.add_argument('--grad_acc_step', type = int, default = 1)
    args = parser.parse_args()
    args.world_size = world_size

    # build model
    # model_path = './config/gpt2'
    # model_path = './config/qwen2.5-instruct'
    model = build_model(args.model, load_weights = args.load, use_lora = args.lora)
    # print(model)
    
    max_step, max_time, bs, seq_len, grad_acc_step = (
        args.max_step, args.max_time, args.bs, args.seq_len, args.grad_acc_step
    )
    tr_out = train(
        model, 
        max_step = max_step, 
        max_time = max_time, 
        bs = bs, 
        seq_len = seq_len, 
        grad_acc_step = grad_acc_step
    )

    records = log_save_record(args, tr_out['step'], tr_out['time'])
    
    

if __name__ == '__main__':
    main()