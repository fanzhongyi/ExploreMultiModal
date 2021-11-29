import os

import pyarrow as pa
import torch

from .base_dataset import BaseDataset


class BookCorpusDataset(BaseDataset):

    def __init__(
        self,
        data_dir: str,
        split="",
        transform=None,
        text_column_name: str = "text",
        max_text_len=196,
        tokenizer=None,
        mlm_collator=None,
        **kwargs,
    ):

        assert split in ["train", "val", "test"]

        if split == "train":
            names = [f"bookcorpus_wikipedia_train_{i}" for i in range(7)]
            names = ["book_wiki_train"]
        elif split == "val":
            names = [
                "bookcorpus_wikipedia_val_0", "bookcorpus_wikipedia_test_0"
            ]
            names = ["book_wiki_val", "book_wiki_test"]
        elif split == "test":
            names = ["bookcorpus_wikipedia_test_0"]
            names = ["book_wiki_test"]
        else:
            names = []

        self.names = names
        self.data_dir = data_dir
        self.transform = transform

        self.text_column_name = text_column_name
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        self.mlm_collator = mlm_collator

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]
            self.table = pa.concat_tables(tables, promote=True)

    def __len__(self):
        return len(self.table)

    def get_text(self, raw_index):
        text = self.table[self.text_column_name][raw_index].as_py()
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


if __name__ == "__main__":
    from transformers import (BertTokenizer, BertTokenizerFast,
                              DataCollatorForLanguageModeling,
                              DataCollatorForWholeWordMask)

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir='/home/babyfan/.cache',
    )

    collator = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                            mlm=True,
                                            mlm_probability=0.40)

    bookcorpus = BookCorpusDataset(data_dir='/home/babyfan/data/arrows',
                                   split='train',
                                   tokenizer=tokenizer,
                                   mlm_collator=collator)
    print(len(bookcorpus))
    __import__('ipdb').set_trace()
