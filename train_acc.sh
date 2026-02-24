### multi gpu ###
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=2 --use_deepspeed finetune_distributed.py 

### single gpu ###
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=1 --main_process_port 28500 --use_deepspeed train_acc.py 