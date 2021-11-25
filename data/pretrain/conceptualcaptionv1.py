#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 babyfan <fanzhongyi1@jd.com>
#
# Distributed under terms of the MIT license.
"""
Conceptual Caption Loader implemented in DALI Framework
"""

import math
import os
import random
import warnings
from timeit import default_timer as timer

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from models.clip import tokenize

warnings.filterwarnings("ignore", ("reader_name"), Warning)


class CaptionDataset:
    def __init__(self, data_root='./datasets', split='val.csv'):
        self.data_root = data_root
        df = pd.read_csv(os.path.join(self.data_root, split), sep='\t')
        img_paths = df['filepath'].tolist()
        caps = df['title'].tolist()
        self.files = list(zip(img_paths, caps))[:20]

    def __getitem__(self, sample_idx):
        filename, caption = self.files[sample_idx]
        cap = tokenize(caption)
        f = open(os.path.join(self.data_root, filename), 'rb')
        img = np.frombuffer(f.read(), dtype=np.uint8)
        return img, cap

    def __len__(self):
        return len(self.files)


class CaptionCallableSource:
    def __init__(
        self,
        batch_size,
        data_root='./datasets',
        split='val.csv',
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=0,
    ):
        self.dataset = CaptionDataset(data_root, split)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.num_replicas = num_replicas
        self.rank = rank
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.reset()

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        sample_idx %= len(self.indices)
        img, cap = self.dataset[self.indices[sample_idx]]
        cap = np.array([sample_idx, self.indices[sample_idx], random.randint(0, len(self.indices)), self.epoch])
        return img, cap

    def __len__(self):
        return self.num_samples

    @property
    def total_size(self):
        return len(self.dataset)

    def reset(self):
        #  self.epoch += 1
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)
        indices = indices[self.rank::self.num_replicas]
        self.indices = indices

    def set_epoch(self, epoch):
        self.epoch = epoch


class CaptionLoader(DALIGenericIterator):
    def __init__(
        self,
        # source params
        batch_size,
        data_root='./datasets',
        is_training=True,
        num_replicas=2,
        rank=0,
        # pipline params
        device_id=0,
        num_threads=4,
        py_num_workers=4,
        py_start_method='spawn',
        # loader params
        output_map=['img', 'cap'],
        auto_reset=True,
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.FILL,
        prepare_first_batch=False,
        # random params
        shuffle=False,
        seed=1,
        # augment params
        resize_shorter=256,
        crop_size=224,
    ):
        source = CaptionCallableSource(
            batch_size,
            data_root=data_root,
            split='train.csv' if is_training else 'val.csv',
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed)

        pipe = Pipeline(batch_size=batch_size,
                        num_threads=num_threads,
                        device_id=device_id,
                        seed=seed,
                        prefetch_queue_depth=1,
                        py_num_workers=py_num_workers,
                        py_start_method=py_start_method)
        with pipe:
            coin = fn.random.coin_flip(probability=0.5)
            imgs, caps = fn.external_source(source=source,
                                            num_outputs=2,
                                            batch=False,
                                            parallel=True,
                                            prefetch_queue_depth=1)
            decode = fn.decoders.image(imgs,
                                       device='mixed',
                                       output_type=types.RGB)
            if is_training:
                imgs = fn.random_resized_crop(
                    decode,
                    size=crop_size,
                    random_area=[0.08, 1.25],
                )
                imgs = fn.crop_mirror_normalize(
                    imgs,
                    mirror=coin,
                    dtype=types.FLOAT,
                    output_layout=types.NCHW,
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                )
            else:
                imgs = fn.resize(
                    decode,
                    resize_shorter=resize_shorter,
                    interp_type=types.INTERP_TRIANGULAR,
                )
                imgs = fn.crop_mirror_normalize(
                    imgs,
                    dtype=types.FLOAT,
                    output_layout=types.NCHW,
                    crop=(crop_size, crop_size),
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                )
            caps.gpu()

            pipe.set_outputs(imgs, caps)
        pipe.build()

        self.pipe = pipe
        self.source = source

        super().__init__(
            pipe,
            size=len(source),
            output_map=output_map,
            auto_reset=auto_reset,
            last_batch_padded=last_batch_padded,
            last_batch_policy=last_batch_policy,
            prepare_first_batch=prepare_first_batch,
        )

    def __next__(self):
        batch = super().__next__()[0]
        img = batch['img']
        cap = batch['cap']
        return img, cap

    def reset(self):
        self.source.reset()
        super().reset()


if __name__ == "__main__":
    loader = CaptionLoader(
        batch_size=4,
        shuffle=False,
        is_training=True,
        last_batch_padded=False,
        last_batch_policy=LastBatchPolicy.FILL,
        num_threads=3,
        py_num_workers=3,
        device_id=0,
        num_replicas=1,
        rank=0,
        prepare_first_batch=False,
    )
    #  for e in range(300):
        #  for i, (img, cap) in enumerate(loader):
            #  img_size = img.size()
            #  cap_size = cap.size()
    epoch = 3
    t_start = timer()
    for e in range(epoch):
        loader.source.set_epoch(e)
        print(f'epoch {e}')
        for i, (img, cap) in enumerate(loader):
            img_size = img.size()
            cap_size = cap.size()
            print(f'iteration {i}', cap)
            #  print(f'iteration {i}', img.size(), cap.size())
    t_cost = timer() - t_start
    #  __import__('ipdb').set_trace()
    print(
        f'time: {t_cost}, speed: {len(loader.source) * epoch / t_cost}imgs/s')
