from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import Block

from models.lib.utils import MaskShuffle, UnMaskShuffle
from models.lib.utils import \
    get_sinusoid_encoding_table_v2 as get_sinusoid_encoding_table


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed, pos_embed="cosine", norm_layer=nn.LayerNorm, act_layer=nn.GELU, pool='mean',
                 classification=False, vit_type="encoder", mask_ratio=0.75, MAE=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            pos_embed (nn.Module): position embedding layer cosine or learnable parameters
            norm_layer: (nn.Module): normalization layer
            pool: 'mean' or 'cls' for classification
            classification: True or False 
            vit_type: "encoder" or "decoder" for MAE
            mask_ratio: a ratio for mask patch numbers
            MAE: Use MAE for trainig 
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1  
        self.classification = classification 
        self.mask_ratio = mask_ratio 
        self.vit_type = vit_type 
        self.MAE = MAE 
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
    
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))
        if pos_embed == "cosine":
            self.register_buffer('pos_embed', get_sinusoid_encoding_table(num_patches + self.num_tokens, embed_dim))
        else:
            self.pos_embed = nn.parameter.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.vit_type == "encoder":
            self.mask_shuffle = MaskShuffle(mask_ratio)
        elif self.vit_type == "decoder":
            self.unmask_embed = UnMaskShuffle(img_size, patch_size, in_chans, embed_dim)
        
        dpr = [x.item() if not self.MAE else 0.0 for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pool = pool
        
        if self.classification:
            self.class_head = nn.Linear(self.num_features, self.num_classes)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def encode(self, x):
        """encoder the no mask patch embedding with position embedding
        Returns:
            norm_embedding: encoder embedding
            sample_index:   a list of token used for encoder
            mask_index:     a list of token mask
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1) 
        x = x + self.pos_embed
        mask_patch_embedding, sample_index, mask_index = self.mask_shuffle(x)
        x = self.blocks(mask_patch_embedding)
        norm_embedding = self.norm(x)
        return norm_embedding, sample_index, mask_index
    
    def decode(self, x, sample_index):
        """decoder the all patch embedding with the mask and position embedding 
        """
        decoder_embed = self.unmask_embed(x, sample_index)
        x = decoder_embed + self.pos_embed 
        x = self.blocks(x)
        return x
    
    def forward_features(self, x):
        """Return the layernormalization features
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x 

    def forward(self, x):
        x = self.forward_features(x)
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise ValueError("pool must be 'cls' or 'mean' ")
        assert x.shape[1] == self.num_features, "outputs must be same with the features"
        if self.classification:
            x = self.class_head(x)
        return x


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_large_patch16_224_decoder(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=512, depth=8, num_heads=16, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


if __name__ == '__main__':
    model = vit_large_patch16_224()
    inputs = torch.randn(5, 3, 224, 224)
    model.cuda()
    inputs = inputs.cuda()
    outputs = model(inputs)
    print(model)
    print(outputs.shape)
    
