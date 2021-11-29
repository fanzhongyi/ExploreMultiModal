import json
import os
from collections import defaultdict

from data.datasets import VQAv2Dataset

from .datamodule_base import BaseDataModule


class VQAv2DataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAv2Dataset

    @property
    def dataset_name(self):
        return "vqa"

    def setup(self):
        super().setup()

        save_path = 'resource'

        if os.path.isfile(f'{save_path}/vqa_dict.json'):

            with open(f'{save_path}/vqa_dict.json') as f:
                vqa_dict = json.load(f)
            self.answer2id = vqa_dict['answer2id']
            self.id2answer = vqa_dict['id2answer']
            self.num_class = vqa_dict['num_class']

        else:

            os.makedirs(save_path, exist_ok=True)

            train_answers = self.train_dataset.table["answers"].to_pandas(
            ).tolist()
            val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
            train_labels = self.train_dataset.table["answer_labels"].to_pandas(
            ).tolist()
            val_labels = self.val_dataset.table["answer_labels"].to_pandas(
            ).tolist()

            all_answers = [
                c for c in train_answers + val_answers if c is not None
            ]
            all_answers = [l for lll in all_answers for ll in lll for l in ll]
            all_labels = [c for c in train_labels + val_labels if c is not None]
            all_labels = [l for lll in all_labels for ll in lll for l in ll]

            self.answer2id = {
                k: int(v) for k, v in zip(all_answers, all_labels)
            }
            sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
            self.num_class = int(max(self.answer2id.values()) + 1)
            self.id2answer = defaultdict(lambda: "unknown")
            for k, v in sorted_a2i:
                self.id2answer[int(v)] = k

            with open(f'{save_path}/vqa_dict.json', 'w') as fw:
                json.dump(dict(answer2id=self.answer2id,
                               id2answer=self.id2answer,
                               num_class=self.num_class),
                          fw,
                          indent=2)

        self.train_dataset.answer2id = self.answer2id
        self.val_dataset.answer2id = self.answer2id
        self.test_dataset.answer2id = self.answer2id

        self.train_dataset.id2answer = self.id2answer
        self.val_dataset.id2answer = self.id2answer
        self.test_dataset.id2answer = self.id2answer

        self.train_dataset.num_class = self.num_class
        self.val_dataset.num_class = self.num_class
        self.test_dataset.num_class = self.num_class
