"""Benchmark GPU Inference Speed of Transformer model. Single GPU"""
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
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from accelerate import Accelerator


    
def get_mem_by_id(index):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.used / 1024**3

def build_model_from_config(model_path):
    # load config from local file
    logging.info('load config')
    config = AutoConfig.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # initialize model
    logging.info('Initialize model.')
    model = AutoModelForCausalLM.from_config(config)
    return model

def build_model_from_weights(model_path, quant = False):
    logging.info('load model')
    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kws['quantization_config'] = quant_config
    else:
        kws = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype = torch.bfloat16, device_map = {'':0}, **kws
    )
    return model

def build_model(
    model_path = './config/qwen2.5-instruct',
    load_weights = False
):
    """
    Build model and place it on correct devices.
    """
    # build model with random initialization or loading weights
    if not load_weights:
        model = build_model_from_config(model_path)
    else:
        model = build_model_from_weights(model_path)
    
    # print number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total params: {total_params / 1000**3:.4f}B')
    
    # set dtype and move to GPU
    logging.debug('Move model to GPU')
    model.to(dtype = torch.bfloat16, device = "cuda:0")
    logging.info(f'Device: {model.device}, dtype: {model.dtype}')

    return model

def infer(
    model,
    max_step, 
    max_time,
    bs,
    src_len,
    gen_len,
):
    """
    Run inference on the model. with sudo input."""

    # prepare optimizer and model
    model.eval()

    # prepare sudo input
    input_ids = torch.randint(1000, 10000, (bs, src_len)).cuda(0)
    
    # run the first inference
    with torch.no_grad():
        _ = model(input_ids)

    start_time = time.time()
    logging.info('Start infering ...')
    tbar = tqdm(
        range(max_step), 
        ncols = 80, 
    )
    for step in tbar:
        with torch.no_grad():
            # generate, force to generate gen_len tokens. Handle the stop creiterion to avoid early stop
            _ = model.generate(input_ids, 
                               max_new_tokens = gen_len,
                               min_new_tokens = gen_len,
                               do_sample = False,
                               pad_token_id = 0,
                               )
        # check stop for max_time
        cur_time = time.time()
        to_stop = cur_time - start_time > max_time
        if to_stop:
            tbar.close()
            break
    tbar.close()

    return {'step': step + 1, 'time': cur_time - start_time}

def calculate_speed(step, time, bs, gen_len):
    return bs * gen_len * step / time

def log_save_record(args, step, tr_time):
    """Get the test record, log and save it"""
    # copy the args
    record = {**args.__dict__}
    
    # add training records
    mem_usage = get_mem_by_id(0)
    speed = calculate_speed(
            step, tr_time, args.bs, args.gen_len
        )
    record.update({'step': step,
                   'infer_time': tr_time,
                   'token/s': speed,
                   'mem_usage': mem_usage})
    
    logging.info(repr(record))

    Path('./results').mkdir(exist_ok = True)
    ori_m = args.model.lower()
    mn = 'gpt2' if 'gpt2' in ori_m else 'qwen2' if 'qwen2' in ori_m else ''
    save_name = f'infer_{mn}_{time.strftime("%Y%m%d_%H%M")}'
    with open(f'./results/{save_name}', 'w') as f:
        json.dump(record, f, indent=  4)

def main():
    # prepare env

    # Set logging format
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                      help = 'model path, e.g., Qwen/Qwen2.5-7B-Instruct', 
                      default = './config/qwen2.5-instruct')
    parser.add_argument('--load', action = 'store_true')
    
    parser.add_argument('--max_step', type = int, default = 1024)
    parser.add_argument('--max_time', type = int, default = 20)
    parser.add_argument('--bs', type = int, default = 1)
    parser.add_argument('--src_len', type = int, default = 512)
    parser.add_argument('--gen_len', type = int, default = 128)
    
    args = parser.parse_args()

    model = build_model(args.model, load_weights = args.load)
    # print(model)
    
    max_step, max_time, bs, src_len, gen_len = (
        args.max_step, args.max_time, args.bs, args.src_len, args.gen_len
    )
    te_out = infer(
        model, 
        max_step = max_step, 
        max_time = max_time, 
        bs = bs, 
        src_len = src_len,
        gen_len = gen_len, 
    )

    records = log_save_record(args, te_out['step'], te_out['time'])
    
    

if __name__ == '__main__':
    main()