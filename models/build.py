from .clip_model import CLIP
from .mae import MAEVisionTransformers


def build_model(config):
    model_type = config.model.type
    if model_type == 'CLIP':
        model = CLIP(
            embed_dim=config.model.clip.embed_dim,
            image_resolution=config.model.clip.image_resolution,
            vision_layers=config.model.clip.vision_layers,
            vision_width=config.model.clip.vision_width,
            vision_patch_size=config.model.clip.vision_patch_size,
            context_length=config.model.clip.context_length,
            vocab_size=config.model.clip.vocab_size,
            transformer_width=config.model.clip.transformer_width,
            transformer_heads=config.model.clip.transformer_heads,
            transformer_layers=config.model.clip.transformer_layers,
        )
    elif model_type == 'EMM':
        # TODO: Implement Model Arch
        model = MAEVisionTransformers()
    elif model_type == 'MAE-v1':
        # NOTE: MAE shuffle mask token into Encoder.
        model = MAEVisionTransformers(
            img_size=config.model.img_size,
            patch_size=16,
            encoder_dim=1024,
            encoder_depth=24,
            encoder_heads=16,
            decoder_dim=512,
            decoder_depth=8,
            decoder_heads=16,
            mask_ratio=0.75,
            flag=0,
        )
    elif model_type == "MAE":
        # NOTE: MAE serialize mask toekn into Encoder.
        model = MAEVisionTransformers()
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
