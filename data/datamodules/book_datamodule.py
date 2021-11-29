from data.datasets import BookDataset

from .datamodule_base import BaseDataModule


class BookDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return BookDataset

    @property
    def dataset_name(self):
        return "book"
