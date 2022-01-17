import os

import torch
import torch.distributed as dist
from data.randaugment import RandomAugment
from PIL import Image
from torchvision import transforms
from transformers import (BertTokenizer, DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask)


class BaseDataModule:

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data.data_root

        self.max_text_len = config.model.max_text_len
        self.image_only = config.data.image_only

        self.tokenizer = self.get_pretrained_tokenizer(config.data.tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        self.image_transforms = ImageTransforms(config)

        if 'pretrain' in config.train.phase:
            train_transform = self.image_transforms.pretrain_transform
        else:
            train_transform = self.image_transforms.train_transform
        self.train_transform = train_transform
        self.val_transform = self.image_transforms.val_transform
        self.test_transform = self.val_transform

        collator = (DataCollatorForWholeWordMask
                    if config.data.whole_word_masking else
                    DataCollatorForLanguageModeling)

        self.mlm_collator = collator(tokenizer=self.tokenizer,
                                     mlm=True,
                                     mlm_probability=config.data.mlm_prob)
        self.image_mask_generator = None

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            transform=self.train_transform,
            split="train",
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            mlm_collator=self.mlm_collator,
            image_mask_generator=self.image_mask_generator,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            transform=self.val_transform,
            split="val",
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            mlm_collator=self.mlm_collator,
            image_mask_generator=self.image_mask_generator,
            image_only=self.image_only,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            transform=self.test_transform,
            split="test",
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            mlm_collator=self.mlm_collator,
            image_mask_generator=self.image_mask_generator,
            image_only=self.image_only,
        )

    def setup(self):
        self.set_train_dataset()
        self.set_val_dataset()
        self.set_test_dataset()

        self.train_dataset.tokenizer = self.tokenizer
        self.val_dataset.tokenizer = self.tokenizer
        self.test_dataset.tokenizer = self.tokenizer

    def get_pretrained_tokenizer(self, from_pretrained):
        tokenizer_dir = f'resource/{from_pretrained}'
        tokenizer_cache_dir = tokenizer_dir + '.cache'

        if os.path.exists(tokenizer_dir):
            return BertTokenizer.from_pretrained(tokenizer_dir,
                                                 cache_dir=tokenizer_cache_dir)

        if dist.is_initialized():
            if dist.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained,
                    do_lower_case="uncased" in from_pretrained,
                    cache_dir=tokenizer_cache_dir,
                ).save_pretrained(tokenizer_dir)
            dist.barrier()

        return BertTokenizer.from_pretrained(
            from_pretrained,
            do_lower_case="uncased" in from_pretrained,
            cache_dir=tokenizer_cache_dir,
        )


class MaskGenerator:

    def __init__(self, input_size, mask_ratio):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        return f"mask {self.num_mask} / total {self.num_patches} patches"

    def __call__(self):
        mask = torch.randperm(self.num_patches) < self.num_mask
        return mask.byte()


class ImageTransforms:

    def __init__(self, config):

        img_size = config.data.img_size
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711))

        self.pretrain_transform = transforms.Compose([
            RandomAugment(2,
                          7,
                          isPIL=True,
                          augs=[
                              'Identity', 'AutoContrast', 'Equalize',
                              'Brightness', 'Sharpness', 'ShearX', 'ShearY',
                              'TranslateX', 'TranslateY', 'Rotate'
                          ]),
            transforms.RandomResizedCrop(img_size,
                                         scale=(0.9, 1.0),
                                         interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_transform = transforms.Compose([
            RandomAugment(2,
                          7,
                          isPIL=True,
                          augs=[
                              'Identity', 'AutoContrast', 'Equalize',
                              'Brightness', 'Sharpness', 'ShearX', 'ShearY',
                              'TranslateX', 'TranslateY', 'Rotate'
                          ]),
            transforms.RandomResizedCrop(img_size,
                                         scale=(0.9, 1.0),
                                         interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
