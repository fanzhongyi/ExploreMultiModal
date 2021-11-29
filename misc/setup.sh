#! /bin/sh
#
# setup.sh
# Copyright (C) 2021 babyfan <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.
#

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r ./requirements.txt

# install dali
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# install apex
git clone https://github.com/NVIDIA/apex
cd apex; pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

