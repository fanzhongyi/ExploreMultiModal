from .book_datamodule import BookDataModule
from .bookcorpus_datamodule import BookCorpusDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .nlvr2_datamodule import NLVR2DataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .wiki_datamodule import WikiDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "book": BookDataModule,
    "wiki": WikiDataModule,
    "bc": BookCorpusDataModule,
}
