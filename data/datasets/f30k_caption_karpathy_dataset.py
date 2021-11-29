from .base_dataset import BaseDataset


class F30KCaptionKarpathyDataset(BaseDataset):

    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
            # names = ["f30k_caption_karpathy_val"]
        elif split == "val":
            names = ["f30k_caption_karpathy_test"]
        elif split == "test":
            names = ["f30k_caption_karpathy_test"]
        else:
            names = []

        super().__init__(*args,
                         **kwargs,
                         names=names,
                         text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)


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

    f30k = F30KCaptionKarpathyDataset(data_dir='/home/babyfan/data/arrows',
                                      split='train',
                                      tokenizer=tokenizer,
                                      mlm_collator=collator)
    print(len(f30k))
    __import__('ipdb').set_trace()
