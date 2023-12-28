  srun  -p pat_rd  -n1 -N1 --gres=gpu:8 \
  accelerate launch --config_file  /mnt/lustre/shanhang/.cache/huggingface/accelerate/default_config.yaml  --main_process_port 29512 \
    llm_infer.py \