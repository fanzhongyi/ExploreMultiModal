#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 babyfan <fanzhongyi1@jd.com>
#
# Distributed under terms of the MIT license.
"""
    This package contains two objects
     - BackgroundGenerator(any_other_generator[,max_prefetch = something])
     - @background([max_prefetch=somethind]) decorator

    the usage is either

    #for batch in BackgroundGenerator(my_minibatch_iterator):
    #    doit()

    or

    #@background()
    #def iterate_minibatches(some_param):
    #    while True:
    #        X = read_heavy_file()
    #        X = do_helluva_math(X)
    #        y = wget_from_pornhub()
    #        do_pretty_much_anything()
    #        yield X_batch, y_batch

    More details are written in the BackgroundGenerator doc
    help(BackgroundGenerator)
"""

import queue
import threading

import torch
from torch.utils.data import DataLoader


class BackgroundGenerator(threading.Thread):
    """
    the usage is below

    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)

    """

    def __init__(self, generator, local_rank=None, max_prefetch=6) -> None:
        """
            This function transforms generator into a background-thead generator.

        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        if self.local_rank:
            # NOTE: easy to switch between (multi-)cuda-related tasks and non-cuda tasks
            torch.cuda.set_device(self.local_rank)

        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, max_prefetch=10, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank
        self.max_prefetch = max_prefetch

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter,
                                        self.local_rank,
                                        max_prefetch=self.max_prefetch)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            if isinstance(self.batch, dict):
                for k in self.batch:
                    self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                     non_blocking=True)
            else:
                for k in range(len(self.batch)):
                    self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                     non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            return
        self.iter.exit_event.set()
        for _ in self.iter:
            ...
        self.iter.join()

    def shutdown(self):
        self._shutdown_background_thread()


# decorator
class background:

    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):

        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs),
                                       max_prefetch=self.max_prefetch)

        return bg_generator
