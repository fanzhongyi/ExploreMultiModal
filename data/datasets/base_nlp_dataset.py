import random

import torch
from torch.utils.data import Dataset, random_split

from datasets import load_from_disk


class BaseNLPDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split="",
        tokenizer=None,
        mlm_collator=None,
        max_text_len=512,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        self.mlm_collator = mlm_collator

        self.data_dir = data_dir

        dataset = load_from_disk(data_dir)
        len_train = int(0.8 * len(dataset['train']))
        len_val = int(0.1 * len(dataset['train']))
        len_test = len(dataset['train']) - len_train - len_val
        dataset_splits = random_split(dataset['train'],
                                      [len_train, len_val, len_test])

        self.splits_id = {'train': 0, 'val': 1, 'test': 2}
        self.text_data = dataset_splits[
            self.splits_id[split]]  # return Subset object

    def __len__(self):
        return len(self.text_data)

    def get_text(self, raw_index):

        return self.get_text_bucket(raw_index)

        text = self.text_data[raw_index]['text']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )
        return {
            "text_ids": encoding.input_ids.squeeze(0),
            "text_mask": encoding.attention_mask.squeeze(0),
            "raw_index": raw_index,
        }

    def get_text_bucket(self, raw_index):
        text = self.text_data[raw_index]['text']
        init_encoding = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_text_len,
        )

        total_text = [text]
        total_len = len(init_encoding.input_ids)

        cur_index = raw_index
        while total_len < self.max_text_len:
            # cur_index = random.randint(0, len(self.text_data) - 1)
            cur_index = (cur_index + 1) % len(self.text_data)
            cur_text = self.text_data[cur_index]['text']
            cur_encoding = self.tokenizer(
                cur_text,
                padding=False,
                truncation=True,
                max_length=self.max_text_len,
            )
            cur_len = len(cur_encoding.input_ids) - 1
            if total_len + cur_len > self.max_text_len:
                break
            total_text.append(cur_text)
            total_len = total_len + cur_len

        total_text_str = ' [SEP] '.join(total_text)

        total_encoding = self.tokenizer(
            total_text_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt',
        )

        return {
            "text_ids": total_encoding.input_ids.squeeze(0),
            "text_mask": total_encoding.attention_mask.squeeze(0),
            "raw_index": raw_index,
        }

    def get_mlm(self, input_ids):
        mlms = self.mlm_collator([input_ids])
        return {
            'text_ids': input_ids,
            'text_labels': torch.full_like(input_ids, -100),
            'text_ids_mlm': mlms['input_ids'].squeeze(0),
            'text_labels_mlm': mlms['labels'].squeeze(0),
        }

    def __getitem__(self, index):
        ret = dict()
        ret.update(self.get_text(index))
        ret.update(self.get_mlm(ret['text_ids']))
        ret = {k: ret[k] for k in ret if '_index' not in k}
        return ret
