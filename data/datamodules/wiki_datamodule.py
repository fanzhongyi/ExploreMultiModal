from data.datasets import WikiDataset

from .datamodule_base import BaseDataModule


class WikiDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WikiDataset

    @property
    def dataset_name(self):
        return "wiki"
