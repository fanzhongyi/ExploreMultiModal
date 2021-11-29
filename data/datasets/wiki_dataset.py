import os

from .base_nlp_dataset import BaseNLPDataset


class WikiDataset(BaseNLPDataset):

    def __init__(
        self,
        data_dir,
        split,
        tokenizer,
        mlm_collator,
        max_text_len=512,
        **kwargs,
    ):
        data_dir = os.path.join(data_dir, 'wikipedia')
        super().__init__(
            data_dir,
            split,
            tokenizer,
            mlm_collator,
            max_text_len,
            **kwargs,
        )
