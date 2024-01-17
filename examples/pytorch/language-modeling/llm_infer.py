import torch_dipu
import torch
import time
import os
import random
import torch.distributed as dist
from my_fully_sharded_data_parallel import *
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM
)
from accelerate import Accelerator
import time

def setup(rank, world_size, port, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    print("comm using port:", str(port))
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def test_infer(MODEL_PATH, model):

  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  prompt = "I believe the meaning of life is"
  rank = int(os.environ['RANK'])
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(rank)

  accelerator = Accelerator()
  model = accelerator.prepare(model)

  print("------start infer------")
  T1 = time.time()
  attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
  pad_token_id = tokenizer.eos_token_id
  generate_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=100)
  T2 = time.time()
  print('times run:', ((T2 - T1)*1000), flush=True)

  answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  print(answer)

if __name__ == "__main__":
    MODEL_PATH = "/mnt/lustre/share_data/PAT/datasets/llama-65b-hf"
    backend = "nccl"
    rank = int(os.environ['RANK'])
    port = 29512 

    world_size = 8
    torch.cuda.set_device(rank)
    setup(rank, world_size, port, backend)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    test_infer(MODEL_PATH, model)
    cleanup()
