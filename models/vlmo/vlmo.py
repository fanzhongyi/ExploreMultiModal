'''
    NOTE:
        VLMO Backbone, without task specified Head, used only for forward_features.

    TODO:
        1. adapt to nvlr2, done
        2. use token_type_embeddings of transformers
        3. add 2d position embedding option.

    FIXME:
        1. fusion_layer should be treated as a model parameter or a forward parameter ?
        2. make distinguishing processing of two mask or remove img_mask ?

    XXX:
        1. forward visual ffn & text ffn, may be parallelized to some extent.

'''

from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp, PatchEmbed, trunc_normal_
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings,
                                                    BertPooler)


def LayerNorm(normalized_shape,
              eps=1e-5,
              elementwise_affine=True,
              export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class MoMEMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        self.v_mlp = Mlp(in_features, hidden_features, out_features, act_layer,
                         drop)
        self.l_mlp = Mlp(in_features, hidden_features, out_features, act_layer,
                         drop)
        self.vl_mlp = Mlp(in_features, hidden_features, out_features, act_layer,
                          drop)

    def forward(self, x, route='vl'):
        assert route in ['v', 'l', 'vl']
        if route == 'v':
            x = self.v_mlp(x)
        elif route == 'l':
            x = self.l_mlp(x)
        elif route == 'vl':
            x = self.vl_mlp(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias,
                                               requires_grad=False),
                 self.v_bias))
        qkv = nn.functional.linear(input=x,
                                   weight=self.qkv.weight,
                                   bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads,
                          C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoMEMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None, route='vl'):
        _x, attn = self.attn(self.norm1(x), mask=mask)

        if self.gamma_1 is None:
            x = x + self.drop_path(_x)
            x = x + self.drop_path(self.mlp(self.norm2(x), route=route))
        else:
            x = x + self.drop_path(self.gamma_1 * _x)
            x = x + self.drop_path(
                self.gamma_2 * self.mlp(self.norm2(x), route=route))
        return x, attn


class VLMO(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_values=None,
        vocab_size=30000,
        max_text_len=27,
        fusion_layer=3,
    ):
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
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # for img embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.pos_embed = nn.parameter.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # for txt embedding
        self.max_text_len = max_text_len
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * mlp_ratio,
            max_position_embeddings=max_text_len,
            hidden_dropout_prob=drop_rate,
            attention_probs_dropout_prob=drop_rate,
        )
        self.bert_config = bert_config
        self.txt_embeddings = BertEmbeddings(bert_config)

        self.token_type_embeddings = nn.Embedding(2, embed_dim)

        #  provided by BertTokenizer
        #  self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #  self.txt_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_cls_token = nn.parameter.Parameter(torch.zeros(
            1, 1, embed_dim))

        self.fusion_layer = fusion_layer

        # Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # pooler & head
        self.pooler = BertPooler(bert_config)
        self.head = nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.img_cls_token, std=0.02)
        self.apply(self._init_weights)

    def embed_img(self, x, img_masks, img_token_type_idx=1):
        # img_token_type_idx for nlvr2
        x = self.patch_embed(x)
        img_cls_token = self.img_cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((img_cls_token, x), dim=1)
        x = x + self.pos_embed
        x = x + self.token_type_embeddings(
            torch.full_like(img_masks, fill_value=img_token_type_idx))
        return x

    def embed_txt(self, x, txt_masks):
        x = self.txt_embeddings(x)
        x = x + self.token_type_embeddings(torch.zeros_like(txt_masks))
        return x

    def forward_features(
        self,
        img=None,
        txt=None,
        img_masks=None,
        txt_masks=None,
        fusion_layer=None,
        img_token_type_idx=1,
    ):

        # forward_img_features
        if txt is None:
            img_embeds = self.embed_img(img, img_masks, img_token_type_idx)
            x = img_embeds
            for _, blk in enumerate(self.blocks):
                x, _ = blk(x, mask=img_masks, route='v')

            x = self.norm(x)
            return x, img_masks

        # forward_txt_features
        if img is None:
            txt_embeds = self.embed_txt(txt, txt_masks)
            x = txt_embeds
            for _, blk in enumerate(self.blocks):
                x, _ = blk(x, mask=txt_masks, route='l')

            x = self.norm(x)
            return x, txt_masks

        # forward txt & img features
        # [T_CLS], w_i, ..., [T_SEP], [PAD], ... | [I_CLS], v_i, ..., [PAD]
        img_embeds = self.embed_img(img, img_masks, img_token_type_idx)
        txt_embeds = self.embed_txt(txt, txt_masks)

        fusion_layer = fusion_layer or self.fusion_layer
        assert 0 <= fusion_layer <= self.bert_config.num_hidden_layers

        for blk in self.blocks[:fusion_layer]:
            img_embeds, _ = blk(img_embeds, mask=img_masks, route='v')
            txt_embeds, _ = blk(txt_embeds, mask=txt_masks, route='l')

        co_embeds = torch.cat([txt_embeds, img_embeds], dim=1)
        co_masks = torch.cat([txt_masks, img_masks], dim=1)

        x = co_embeds
        for blk in self.blocks[fusion_layer:]:
            x, _ = blk(x, mask=co_masks, route='vl')

        x = self.norm(x)
        return x, co_masks

    def forward(
        self,
        img=None,
        txt=None,
        img_masks=None,
        txt_masks=None,
        fusion_layer=None,
        img_token_type_idx=1,
    ):
        x, x_mask = self.forward_features(
            img=img,
            txt=txt,
            img_masks=img_masks,
            txt_masks=txt_masks,
            fusion_layer=fusion_layer,
            img_token_type_idx=img_token_type_idx,
        )
        x = x[:, 0]
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "img_cls_token"}

    # NOTE: token from MoCoV3, maybe needed for 2D pos_embed
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                            dim=1)[None, :, :]
        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.parameter.Parameter(
            torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


if __name__ == "__main__":
    from pprint import pprint
    from time import time

    from transformers import BertTokenizer, DataCollatorForWholeWordMask

    txt = [
        'I like china',
        'I will win the 2021 VisualQA Challenge, yes',
        'I have two week to build my baseline model',
    ]
    s_time = time()
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', cache_dir='resource/tokenizer_cache')
    tokenizer.save_pretrained('resource/tokenizer')
    tokenizer = BertTokenizer.from_pretrained(
        'resource/tokenizer', cache_dir='resource/tokenizer_cache')

    e_time = time()
    print(e_time - s_time)
    txt = tokenizer(txt,
                    padding='max_length',
                    truncation=True,
                    max_length=27,
                    return_tensors='pt')
    pprint(txt.attention_mask.size())
    pprint(txt.input_ids.size())
    pprint(txt.input_ids)
    pprint(txt.attention_mask)
    exit(0)

    mask_engine = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                               mlm=True,
                                               mlm_probability=0.40)

    out = mask_engine(list(txt.input_ids))
    pprint(out)

    img = torch.ones([3, 3, 224, 224])
    vlmo = VLMO(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        vocab_size=30000,
        max_text_len=27,
        fusion_layer=3,
    )
    n_params = sum(p.numel() for p in vlmo.parameters() if p.requires_grad)
    print(f"number of params: {n_params / 1e6} M")

    img_feature = vlmo(
        img=img,
        img_masks=torch.ones([3, 197], dtype=torch.int),
    )
    print(img_feature.size())

    txt_feature = vlmo(
        txt=txt['input_ids'],
        txt_masks=txt['attention_mask'],
    )
    print(txt_feature.size())

    res = vlmo.forward_features(
        img=img,
        img_masks=torch.ones([3, 197], dtype=torch.int),
        txt=txt['input_ids'],
        txt_masks=txt['attention_mask'],
        fusion_layer=5,
    )
    print(type(res[0]))
    print(res[0].size())

    print(type(res[1]))
    print(res[1].size())

    res = vlmo.forward_features(
        txt=txt['input_ids'],
        txt_masks=txt['attention_mask'],
    )
    print(type(res[0]))
    print(res[0].size())

    print(type(res[1]))
    print(res[1].size())

    res = vlmo.forward_features(
        img=img,
        img_masks=torch.ones([3, 197], dtype=torch.int),
    )
    print(type(res[0]))
    print(res[0].size())

    print(type(res[1]))
    print(res[1].size())
