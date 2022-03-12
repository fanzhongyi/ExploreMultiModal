import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        names: list,
        transform=None,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        tokenizer=None,
        mlm_collator=None,
        image_mask_generator=None,
        image_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()

        self.names = names
        self.data_dir = data_dir
        self.transform = transform

        self.text_column_name = text_column_name
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        self.mlm_collator = mlm_collator

        self.image_mask_generator = image_mask_generator
        self.image_only = image_only

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas(
                ).tolist()
                self.all_texts = ([
                    list(set(texts)) for texts in self.all_texts
                ] if remove_duplicate else self.all_texts)
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_image(self, index, image_key="image"):
        index, _ = self.index_mapper[index]
        image_bytes = self.table[image_key][index].as_py()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image_aug = self.transform(image)[0]
            image = self.transform(image)

        if isinstance(image, tuple):
            image, image4dalle = image[0], image[1]
        else:
            image4dalle = image

        return {
            "image": image,
            "image_aug": image_aug,
            "image4dalle": image4dalle,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]
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
            "img_index": index,
            "cap_index": caption_index,
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

    def get_mim(self, raw_image):
        return {
            'image_bool_masked_pos': self.image_mask_generator(),
        }

    def get_suite(self, index):
        ret = dict()

        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                ret.update(self.get_text(index))
                ret.update(self.get_mlm(ret['text_ids']))
                ret.update(self.get_mim(ret['image']))

                ret = {k: ret[k] for k in ret if '_index' not in k}

                result = True
            except Exception as e:
                print(f"Error read idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        '''
        ret keys:
            - image, text_ids, text_mask,
            - img_index, cap_index, raw_index,
            - text_ids, text_mask, text_labels, text_ids_mlm, text_labels_mlm,
            - image, image_mask,
        '''
        return ret

    def __getitem__(self, index):
        assert index
        raise NotImplementedError
