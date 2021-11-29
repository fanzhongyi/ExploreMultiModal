from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from .bg_dataloader import DataLoaderX
from .datamodules import _datamodules


class MTDataModule:
    ''' Multi-Task DataModule:
        Encapsulate multi dataset and provide task-specific Dataloader.
        e.g.
            1. Dataset(CC3M, SBU, Coco, VG) -> pretrain_mum task
            2. Dataset(VQAv2) -> finetune_vqa task
            3. ...
    '''

    def __init__(self, config):
        dm_keys = config.train.datasets
        assert len(dm_keys) > 0

        self.dm_keys = dm_keys
        self.dm_dicts = {key: _datamodules[key](config) for key in dm_keys}
        self.dms = [v for _, v in self.dm_dicts.items()]

        self.phase = config.train.phase
        self.train_dispatch = DispatchWarpper(self.phase, 'train')
        self.val_dispatch = DispatchWarpper(self.phase, 'val')
        self.test_dispatch = DispatchWarpper(self.phase, 'test')

        self.batch_size = config.data.batch_size
        self.eval_batch_size = config.data.eval_batch_size or self.batch_size * 2

        self.bg_loader = config.data.bg_loader

        self.collate = None

        self.vocab_size = config.model.vocab_size

        self.num_workers = config.data.py_num_workers
        self.prefetch_queue_depth = config.data.prefetch_queue_depth
        self.prefetch_factor = config.data.prefetch_factor

        self.world_size = config.dist.world_size
        self.rank = config.dist.rank
        self.local_rank = config.dist.local_rank

        self.seed = config.seed

        self.setup()

    def setup(self):
        for dm in self.dms:
            dm.setup()

        self.train_dataset = ConcatDataset(
            [dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])

        self.tokenizer = self.dms[0].tokenizer

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.test_sampler = DistributedSampler(
            self.test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

    def train_dataloader(self):
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
        loader = self.get_dataloader(kwargs)
        return loader

    def val_dataloader(self):
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
        loader = self.get_dataloader(kwargs)
        return loader

    def test_dataloader(self):
        kwargs = dict(
            dataset=self.test_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
        loader = self.get_dataloader(kwargs)
        return loader

    def get_dataloader(self, kwargs):
        if self.bg_loader:
            return DataLoaderX(local_rank=self.local_rank,
                               max_prefetch=self.prefetch_queue_depth,
                               **kwargs)
        else:
            return DataLoader(**kwargs)

    # def collate(self, batch):
    #     batch_size = len(batch)
    #     ...


class DispatchWarpper():
    ''' Multi-Task dispatch_mapping'''

    def __init__(self, phase, train_phase='train'):
        assert phase in [
            'pretrain_vis',
            'pretrain_txt',
            'pretrain_mum',
            'finetune_vis',
            'finetune_txt',
            'finetune_vqa',
            'finetune_retrieval',
            'finetune_ref',
            'finetune_nlvr2',
            'finetune_caption',
            'finetune_inpainting',
        ]
        self.phase = phase
        self.train_phase = train_phase
        self.dispatch_mapping(self.phase, self.train_phase)

    def __call__(self, sample_dict: dict) -> list:
        assert set(self.output_map) <= set(sample_dict.keys(
        )), f'sample miss keys {set(self.output_map) - set(sample_dict.keys())}'
        sample_tuple = [sample_dict.get(k) for k in self.output_map]
        return sample_tuple

    def dispatch_mapping(self, phase, train_phase):
        if phase in ['pretrain_mum', 'finetune_retrieval', 'finetune_caption']:
            if train_phase in ['train', 'val', 'test']:
                self.output_map = [
                    'image',
                    'text_mask',
                    'text_ids',
                    'text_labels',
                    'text_ids_mlm',
                    'text_labels_mlm',
                ]

        elif phase in ['pretrain_vis']:
            if self.train_phase in ['train', 'val']:
                self.output_map = ['image', 'image_mask']
            else:
                self.output_map = ['image', 'image_mask']

        elif phase in ['finetune_vqa']:
            self.output_map = [
                'image',
                'text_mask',
                'text_ids',
                'vqa_targets',
                'qid',
            ]
        else:
            self.output_map = [
                'image',
                'text_mask',
                'text_ids',
                'text_labels',
                'text_ids_mlm',
                'text_labels_mlm',
            ]
