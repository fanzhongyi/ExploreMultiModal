FROM nvcr.io/nvidia/pytorch:21.12-py3

ENV OMP_NUM_THREADS=20
ENV WANDB_API_KEY=30e859c562557e3cb316b5863156a37c09569611

# basic python packages
RUN pip install torch torchvision einops regex tqdm pandas wandb timm transformers deepspeed pyarrow datasets ftfy hydra-core ipdb -i https://pypi.tuna.tsinghua.edu.cn/simple --proxy="http://39.106.71.115:7890"

# workspace
WORKDIR /ExploreMultiModal


### USAGE:

# docker build -t vlmo/rep:pre .

# docker run -it --rm --ipc=host --net=host --gpus all --ulimit stack=67108864 --name vlmo -v /home/babyfan/ExploreMultiModal:/ExploreMultiModal -v /home/babyfan/dataset:/ExploreMultiModal/datasets vlmo/rep:pre /bin/bash

#  torchrun --nnodes 1 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 12345 main.py seed=6309 train.epochs=91 train.print_stat_level=2 data.batch_size=180 data.eval_batch_size=100 data.num_threads=4 data.py_num_workers=4 train.warmup_epochs=1 train.warmup_steps=1 throughput_mode=false tag=test train=pretrain_mum log_level=debug ipdb=false model.name='vlmo_large'

# OMP_NUM_THREADS=10 torchrun --nnodes 1 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 12345 main.py seed=6309 train.epochs=91 train.print_stat_level=2 data.batch_size=180 data.eval_batch_size=100 data.num_threads=4 data.py_num_workers=4 train.warmup_epochs=1 train.warmup_steps=1 throughput_mode=false tag=test train=pretrain_mum log_level=debug ipdb=false model.name='vlmo_large'
