import os

from .base_nlp_dataset import BaseNLPDataset


class BookDataset(BaseNLPDataset):

    def __init__(
        self,
        data_dir,
        split,
        tokenizer,
        mlm_collator,
        max_text_len=512,
        **kwargs,
    ):
        data_dir = os.path.join(data_dir, 'bookcorpus')
        super().__init__(
            data_dir,
            split,
            tokenizer,
            mlm_collator,
            max_text_len,
            **kwargs,
        )


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

    book = BookDataset(data_dir='/home/babyfan/data/arrows',
                       split='train',
                       tokenizer=tokenizer,
                       mlm_collator=collator)
    print(len(book))
    __import__('ipdb').set_trace()
