# Benchmark the GPU Performance

This repository provide a benchmark of the GPU performance on deep learning.

Focus on **generative** model training and inference.

We test the following performances:
- Single GPU, Transformer training
  - Parameters: batch_size, seq_len, grad_acc_step?
  - Metrics: tokens/second
- Single GPU, Transformer inference
  - Parameters: batch_size, seq_len
- Multi GPU, Transformer, Distributed Training w/ accelerate
  - Parameters: batch_size, seq_len

Note: `run.py` is the old version

## Train

Test the training speed (tokens/second)

**Test on GPT2**
```Bash
accelerate launch --num_processes 1 bench.py --model ./config/gpt2
accelerate launch --num_processes 2 bench.py --model ./config/gpt2
accelerate launch --num_processes 4 bench.py --model ./config/gpt2
```

**Test on Qwen2.5**

```Bash
# random initialization
accelerate launch --num_processes 1 bench.py --model ./config/qwen2.5-instruct --lora
accelerate launch --num_processes 4 bench.py --model ./config/qwen2.5-instruct --lora

# load weights
accelerate launch --num_processes 1 bench.py --model Qwen/Qwen2.5-7B-Instruct --lora --load
accelerate launch --num_processes 4 bench.py --model Qwen/Qwen2.5-7B-Instruct --lora --load
```

For single GPU, you can also run with
```Bash
# load weights
HF_HUB_CACHE=/next_share/hf_cache/hub/ python bench.py --model Qwen/Qwen2.5-7B-Instruct --load --lora
```

## Infer
```Bash
python bench_infer.py --model ./config/gpt2
python bench_infer.py --model ./config/qwen2.5-instruct
# from local weights
python bench_infer.py --model Qwen/Qwen2.5-7B-Instruct --load
```