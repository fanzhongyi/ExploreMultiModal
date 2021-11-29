import torch

from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):

    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_trainable_val"]
            # names = ["vqav2_rest_val"]
        elif split == "val":
            names = ["vqav2_rest_val"]
        elif split == "test":
            names = ["vqav2_test"]  # vqav2_test-dev for test-dev
            # names = ["vqav2_rest_val"]
        else:
            names = []

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image = self.get_image(index)["image"]
        text_ret = self.get_text(index)
        text_ids = text_ret['text_ids']
        text_mask = text_ret['text_mask']

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        qid = torch.tensor([qid])
        vqa_targets = torch.zeros(self.num_class)

        for label, score in zip(labels, scores):
            vqa_targets[label] = score

        return {
            "image": image,
            "text_ids": text_ids,
            'text_mask': text_mask,
            "vqa_targets": vqa_targets,
            # "vqa_answer": answers,
            # "vqa_labels": labels,
            # "vqa_scores": scores,
            "qid": qid,
        }


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

    vqav2 = VQAv2Dataset(data_dir='/home/babyfan/data/arrows',
                         split='train',
                         tokenizer=tokenizer,
                         mlm_collator=collator)
    vqav2.num_class = 3128
    print(len(vqav2))
    __import__('ipdb').set_trace()
