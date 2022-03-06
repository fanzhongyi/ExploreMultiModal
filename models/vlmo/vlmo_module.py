from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.utils import ModelEmaV2 as ModelEma

from . import objectives
from .heads import ISDAHead, ITCHead, ITMHead, MIMHead, MLMHead, MPPHead
from .vlmo import VLMO, LayerNorm


class VlmoModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        model_cfg = config.model
        norm_layer = partial(LayerNorm,
                             eps=1e-12,
                             export='fused' not in model_cfg.norm_layer)

        self.transformer = VLMO(
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
            in_chans=model_cfg.in_chans,
            num_classes=model_cfg.num_classes,
            embed_dim=model_cfg.embed_dim,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            qkv_bias=model_cfg.qkv_bias,
            qk_scale=None,
            drop_rate=model_cfg.drop_rate,
            attn_drop_rate=model_cfg.attn_drop_rate,
            drop_path_rate=model_cfg.drop_path_rate,
            norm_layer=norm_layer,
            init_values=model_cfg.init_values,
            vocab_size=model_cfg.vocab_size,
            max_text_len=model_cfg.max_text_len,
            fusion_layer=model_cfg.fusion_layer,
        )
        self._freeze_params()

        self.loss_names = config.train.loss_names
        hs = model_cfg.embed_dim

        # ===================== Pretrain ===================== #

        if 'mlm' in self.loss_names:
            self.mlm_head = MLMHead(
                self.transformer.bert_config,
                weight=self.transformer.txt_embeddings.word_embeddings.weight)
            self.mlm_head.apply(self.transformer._init_weights)

        if 'itc' in self.loss_names:
            self.itc_head = ITCHead(hs, model_cfg.itc_dim)
            self.itc_head.apply(self.transformer._init_weights)
            self.itc_temp = nn.Parameter(
                torch.ones([]) * np.log(1 / model_cfg.itc_temp))

        if 'itm' in self.loss_names:
            self.itm_head = ITMHead(hs)
            self.itm_head.apply(self.transformer._init_weights)

        if 'mim' in self.loss_names:
            self.d_vae = objectives.create_d_vae(
                weight_path=config.train.discrete_vae_weight_path,
                d_vae_type=config.train.discrete_vae_type,
                device="cpu",
                image_size=model_cfg.img_size // 2)

            for p in self.d_vae.parameters():
                p.requires_grad = False

            self.mim_head = MIMHead(hs, model_cfg.img_vocab_size)
            self.mim_head.apply(self.transformer._init_weights)

        if 'mpp' in self.loss_names:
            self.mpp_head = MPPHead(self.transformer.bert_config)
            self.mpp_head.apply(self.transformer._init_weights)

        # ==================== Downstream ==================== #

        if 'vqa' in self.loss_names:
            vs = config.data.vqav2_label_size
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                norm_layer(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(self.transformer._init_weights)
            self.vqa_last = None

            if config.train.isda_lambda > 0.:
                self.vqa_last = self.vqa_classifier[-1]
                self.vqa_classifier = self.vqa_classifier[:-1]
                self.isda_head = ISDAHead(hidden_size=hs * 2, class_num=vs)

        if 'nlvr2' in self.loss_names:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                norm_layer(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(self.transformer._init_weights)
            self.transformer.nlvr2_embedding = nn.Embedding(3, hs)

        if 'irtr' in self.loss_names:
            self.margin = 0.2
            self.rank_output = nn.Linear(hs, 1)

        if 'refcoco' in self.loss_names:
            ...

        # ===================== ModelEma ===================== #

        self.transformer_m = None
        if self.config.vlmo_ema:
            self.transformer_m = ModelEma(self.transformer,
                                          decay=self.config.vlmo_ema_decay)
            for p in self.transformer_m.parameters():
                p.requires_grad = False

        # ======================= Queue ====================== #

        if hasattr(config.train, 'neg_queue') and config.train.neg_queue:
            self.q_size = config.train.queue_size
            self.register_buffer("img_queue",
                                 torch.randn(model_cfg.itc_dim, self.q_size))
            self.register_buffer("txt_queue",
                                 torch.randn(model_cfg.itc_dim, self.q_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
            self.txt_queue = nn.functional.normalize(self.txt_queue, dim=0)
        else:
            self.q_size = 0
            self.img_queue, self.txt_queue = None, None

    def _freeze_params(self):

        if self.config.train.phase in ['pretrain_txt']:
            for b in self.transformer.blocks:
                del b.mlp.vl
                if self.config.train.fixed_attn:
                    b.gamma_1.requires_grad = False
                    b.gamma_2.requires_grad = False
                    for p in b.attn.parameters():
                        p.requires_grad = False
                    for p in b.norm1.parameters():
                        p.requires_grad = False
                    for p in b.norm2.parameters():
                        p.requires_grad = False
            for p in self.transformer.norm.parameters():
                p.requires_grad = False

        elif self.config.train.phase in ['pretrain_mum', 'finetune_vqa']:
            for b in self.transformer.blocks[:self.transformer.fusion_layer]:
                del b.mlp.vl

    def _adjust_downstream_params(self):

        if 'nlvr2' in self.loss_names:
            # copy token_type_embeddings params
            emb_data = self.transformer.token_type_embeddings.weight.data
            self.transformer.nlvr2_embedding.weight.data[:-1] = emb_data
            self.transformer.nlvr2_embedding.weight.data[-1] = emb_data[-1]
            self.transformer.token_type_embeddings = self.transformer.nlvr2_embedding

        if 'irtr' in self.loss_names:
            self.rank_output.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_head.fc.bias.data[1:]
            for p in self.itm_head.parameters():
                p.requires_grad = False

        if 'refcoco' in self.loss_names:
            ...

    def interpolate_pos_embedding(self, state_dict):

        pos_embed_keys = ['pos_embed', 'transformer.pos_embed']

        for pos_embed_key in pos_embed_keys:

            if pos_embed_key in state_dict:
                pos_embed_ckpt = state_dict[pos_embed_key]
                embedding_size = pos_embed_ckpt.shape[-1]
                num_patches = self.transformer.patch_embed.num_patches
                num_extra_tokens = self.transformer.pos_embed.shape[
                    -2] - num_patches
                orig_size = int(
                    (pos_embed_ckpt.shape[-2] - num_extra_tokens)**0.5)
                new_size = int(num_patches**0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" %
                          (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_ckpt[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_ckpt[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                    embedding_size).permute(
                                                        0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False,
                    )
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat(
                        (extra_tokens, pos_tokens), dim=1)
                    state_dict[pos_embed_key] = new_pos_embed

        txt_pos_embed_key = 'transformer.txt_embeddings.position_embeddings.weight'
        txt_pos_id_key = 'transformer.txt_embeddings.position_ids'

        if txt_pos_embed_key in state_dict:
            txt_pos_embed_ckpt = state_dict[txt_pos_embed_key]
            state_dict[txt_pos_embed_key] = txt_pos_embed_ckpt[:self.transformer
                                                               .max_text_len, :]
        if txt_pos_id_key in state_dict:
            txt_pos_id_ckpt = state_dict[txt_pos_id_key]
            state_dict[txt_pos_id_key] = txt_pos_id_ckpt[:, :self.transformer.
                                                         max_text_len]

        return state_dict

    def _load_vlmo(self, state_dict):

        for k in list(state_dict.keys()):
            # NOTE: Backward compatibility
            if '.mlp.v_mlp' in k:
                state_dict[k.replace(".mlp.v_mlp", ".mlp.v")] = state_dict[k]
                del state_dict[k]
            if '.mlp.l_mlp' in k:
                state_dict[k.replace(".mlp.l_mlp", ".mlp.l")] = state_dict[k]
                del state_dict[k]
            if 'mlp.vl_mlp' in k:
                state_dict[k.replace(".mlp.vl_mlp", ".mlp.vl")] = state_dict[k]
                del state_dict[k]

            # if k.endswith('gamma_1'):
            #     state_dict[k.replace("gamma_1", "gamma_1.v")] = state_dict[k]
            #     state_dict[k.replace("gamma_1", "gamma_1.l")] = state_dict[k]
            #     state_dict[k.replace("gamma_1", "gamma_1.vl")] = state_dict[k]
            # if k.endswith('gamma_2'):
            #     state_dict[k.replace("gamma_2", "gamma_2.v")] = state_dict[k]
            #     state_dict[k.replace("gamma_2", "gamma_2.l")] = state_dict[k]
            #     state_dict[k.replace("gamma_2", "gamma_2.vl")] = state_dict[k]

        matching = self.load_state_dict(state_dict, strict=False)
        self._adjust_downstream_params()

        return matching

    def _load_beit(self, state_dict):

        for k in list(state_dict.keys()):
            if "mlp" in k:
                state_dict[k.replace(".mlp", ".mlp.v")] = state_dict[k]
                del state_dict[k]

            if 'cls_token' in k:
                state_dict[k.replace("cls_token",
                                     "img_cls_token")] = state_dict[k]
                del state_dict[k]

            if 'mask_token' in k:
                state_dict[k.replace("mask_token",
                                     "img_mask_token")] = state_dict[k]
                del state_dict[k]

            # if 'gamma_1' in k:
            #     state_dict[k.replace("gamma_1", "gamma_1.v")] = state_dict[k]
            #     del state_dict[k]
            # if 'gamma_2' in k:
            #     state_dict[k.replace("gamma_2", "gamma_2.v")] = state_dict[k]
            #     del state_dict[k]

            if 'lm_head' in k:
                state_dict[k.replace("lm_head", "fc")] = state_dict[k]
                del state_dict[k]

        if 'mim' in self.loss_names:
            self.mim_head.load_state_dict(state_dict, strict=False)

        matching = self.transformer.load_state_dict(state_dict, strict=False)
        self._adjust_downstream_params()

        return matching

    def load_from_ckpt(self, state_dict):

        state_dict = self.interpolate_pos_embedding(state_dict)

        is_beit = True
        has_momentum_dict = False
        for k in list(state_dict.keys()):
            if ".mlp.v" in k or '.mlp.l' in k or '.mlp.vl' in k:
                is_beit = False
            if "transformer_m" in k:
                has_momentum_dict = True

        load_fn = self._load_beit if is_beit else self._load_vlmo
        matching = load_fn(state_dict)

        if self.transformer_m is not None and not has_momentum_dict:
            self.transformer_m.set(self.transformer)

        return matching, is_beit

    def infer(
        self,
        batch,
        infer_mode='img-txt',
        mask_txt=False,
        mask_img=False,
        image_token_type_idx=1,
        momentum_mode=False,
    ):
        assert infer_mode in ['img_only', 'txt_only', 'img-txt']

        if momentum_mode:
            assert self.transformer_m is not None
            transformer = self.transformer_m.module
        else:
            transformer = self.transformer

        img, img_attn_masks, bool_masked_pos = None, None, None
        txt_ids, txt_labels, txt_attn_masks = None, None, None

        if 'img' in infer_mode:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"

            img = batch[imgkey]
            B = img.size(0)
            def_attn_masks = torch.ones(
                [B, transformer.patch_embed.num_patches + 1],
                dtype=torch.int64,
                device=img.device)
            # img_attn_masks = batch["image_mask"] if mask_img else def_attn_masks
            img_attn_masks = def_attn_masks
            bool_masked_pos = batch[
                "image_bool_masked_pos"] if mask_img else None

        if 'txt' in infer_mode:
            do_mlm = "_mlm" if mask_txt else ""
            txt_ids = batch[f"text_ids{do_mlm}"]
            txt_labels = batch[f"text_labels{do_mlm}"] if mask_txt else None
            txt_attn_masks = batch["text_mask"]

        co_feats, _ = transformer.forward_features(
            img=img,
            txt=txt_ids,
            img_attn_masks=img_attn_masks,
            txt_attn_masks=txt_attn_masks,
            bool_masked_pos=bool_masked_pos,
            fusion_layer=None,
        )

        if txt_ids is not None:
            txt_feats, img_feats = (co_feats[:, :transformer.max_text_len],
                                    co_feats[:, transformer.max_text_len:])
        else:
            txt_feats, img_feats = None, co_feats

        cls_feats = transformer.pooler(co_feats)

        ret = {
            "txt_feats": txt_feats,
            "img_feats": img_feats,
            "co_feats": co_feats,
            # "cls_feats": co_feats[:, 0],
            "cls_feats": cls_feats,
            "img_masks": img_attn_masks,
            "img_bool_masked_pos": bool_masked_pos,
            "txt_labels": txt_labels,
            "txt_ids": txt_ids,
            "txt_masks": txt_attn_masks,
        }
        return ret

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)

        ret = dict()
        if len(self.loss_names) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.loss_names:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Image Modeling
        if "mim" in self.loss_names:
            ret.update(objectives.compute_mim(self, batch))

        # Image Text Contrast
        if "itc" in self.loss_names:
            ret.update(objectives.compute_itc(self, batch))

        # Image Text Matching
        if "itm" in self.loss_names:
            itc_ret = ret if "itc" in self.loss_names else None
            ret.update(objectives.compute_itm(self, batch, itc_ret))

        # Visual Question Answering
        if "vqa" in self.loss_names:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.loss_names:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.loss_names:
            ret.update(objectives.compute_irtr(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.loss_names:
            ret.update(objectives.compute_mpp(self, batch))

        return ret

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "itc_temp", "transformer.pos_embed", "transformer.img_cls_token"
        }
