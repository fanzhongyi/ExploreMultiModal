from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from timm.data import Mixup

from data.pretrain.conceptualcaption import CaptionLoader


def build_loader(config):

    train_loader = CaptionLoader(
        batch_size=config.data.batch_size,
        data_root=config.data.data_path,
        is_training=True,
        num_replicas=config.dist.world_size,
        rank=config.dist.rank,
        device_id=config.dist.local_rank,
        start_epoch=0,
        num_threads=2,
        py_num_workers=2,
        output_map=('img', 'cap'),
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.FILL,
        shuffle=True,
        seed=config.seed,
    )

    val_loader = CaptionLoader(
        batch_size=config.data.batch_size,
        data_root=config.data.data_path,
        is_training=False,
        num_replicas=config.dist.world_size,
        rank=config.dist.rank,
        device_id=config.dist.local_rank,
        start_epoch=0,
        num_threads=2,
        py_num_workers=2,
        output_map=('img', 'cap'),
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        shuffle=False,
        seed=config.seed,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.data.aug.mixup > 0 or config.data.aug.cutmix > 0. or config.data.aug.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.data.aug.mixup,
                         cutmix_alpha=config.data.aug.cutmix,
                         cutmix_minmax=config.data.aug.cutmix_minmax,
                         prob=config.data.aug.mixup_prob,
                         switch_prob=config.data.aug.mixup_switch_prob,
                         mode=config.data.aug.mixup_mode,
                         label_smoothing=config.model.label_smoothing,
                         num_classes=config.model.num_classes)

    return train_loader, val_loader, mixup_fn
