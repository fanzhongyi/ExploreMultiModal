import glob
import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_vqa_score(logits, target):
    logits = torch.max(logits, 1)[1]
    one_hots = torch.zeros_like(target, device=target.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * target

    count = len(logits)
    mean_score = scores.sum() / count

    return mean_score, count


def compute_accuracy(logits, target):
    """ MLM Adapted Version """
    preds = logits.argmax(dim=-1)
    preds = preds[target != -100]
    target = target[target != -100]
    if target.numel() == 0:
        return torch.tensor(0, device=target.device), 0

    assert preds.shape == target.shape

    mean_acc = (preds == target).float().mean()
    count = target.numel()

    return mean_acc, count


def compute_mlm(model, batch):
    """ support multimodal & txt_only """
    has_img = any(['image' in k for k in batch.keys()])

    infer = model.infer(batch,
                        infer_mode='img-txt' if has_img else 'txt_only',
                        mask_txt=True,
                        mask_img=False)

    txt_feats = infer["txt_feats"]
    mlm_labels = infer["txt_labels"]

    mask = (mlm_labels != -100).unsqueeze(-1).expand_as(txt_feats)
    masked_txt_feats = txt_feats[mask].contiguous().view(-1, txt_feats.size(-1))

    mlm_logits = model.mlm_head(masked_txt_feats)
    mlm_labels = mlm_labels[mlm_labels != -100]

    mlm_mean_acc, mlm_count = compute_accuracy(mlm_logits, mlm_labels)

    if mlm_count > 0:
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, model.config.model.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )
    else:
        mlm_loss = 0.

    ret = {
        "mlm_task_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["txt_ids"],
        "mlm_mean_acc": mlm_mean_acc,
        "mlm_count": mlm_count,
    }
    return ret


def compute_itc(model, batch):
    with torch.no_grad():
        model.itc_temp.data = torch.clamp(model.itc_temp.data, 0, 4.6052)

    temp = model.itc_temp.exp()

    img_infer = model.infer(batch, infer_mode='img_only')
    txt_infer = model.infer(batch, infer_mode='txt_only')

    i_feat = model.itc_head(img_infer['co_feats'][:, 0])
    t_feat = model.itc_head(txt_infer['co_feats'][:, 0])

    if model.transformer_m is not None:
        model.transformer_m.eval()
        with torch.no_grad():
            model.transformer_m.update(model.transformer)

            img_infer_m = model.infer(batch,
                                      infer_mode='img_only',
                                      momentum_mode=True)
            txt_infer_m = model.infer(batch,
                                      infer_mode='txt_only',
                                      momentum_mode=True)

            i_feat_m = model.itc_head(img_infer_m['co_feats'][:, 0])
            t_feat_m = model.itc_head(txt_infer_m['co_feats'][:, 0])
        '''
            sim_i2t_targets = alpha * F.softmax(
                sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(
                sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
        '''

    if model.transformer_m is None and model.img_queue is None:
        sim_i2t = i_feat @ t_feat.t() * temp
        sim_t2i = sim_i2t.t()

    elif model.img_queue is None:
        sim_i2t = i_feat @ t_feat_m.t() * temp
        sim_t2i = t_feat @ i_feat_m.t() * temp

    else:
        i_feat_all = torch.cat(
            [i_feat_m.t(), model.img_queue.clone().detach()], dim=1)
        t_feat_all = torch.cat(
            [t_feat_m.t(), model.txt_queue.clone().detach()], dim=1)

        sim_i2t = i_feat @ t_feat_all * temp
        sim_t2i = t_feat @ i_feat_all * temp

        dequeue_and_enqueue(model, i_feat_m, t_feat_m)

    sim_targets = torch.arange(sim_i2t.size(0), device=sim_i2t.device)
    i2t_loss = F.cross_entropy(sim_i2t, sim_targets)
    t2i_loss = F.cross_entropy(sim_t2i, sim_targets)

    itc_loss = (i2t_loss + t2i_loss) / 2

    bs = sim_i2t.size(0)

    itc_i2t_mean_acc, itc_i2t_count = compute_accuracy(sim_i2t[:, :bs],
                                                       sim_targets)
    itc_t2i_mean_acc, itc_t2i_count = compute_accuracy(sim_t2i[:, :bs],
                                                       sim_targets)

    ret = {
        'itc_task_loss': itc_loss,
        'i2t_Loss': i2t_loss,
        't2i_Loss': t2i_loss,
        'sim_i2t': sim_i2t,
        'sim_t2i': sim_t2i,
        'itc_temp': temp.data,
        'itc_i2t_mean_acc': itc_i2t_mean_acc,
        'itc_i2t_count': itc_i2t_count,
        'itc_t2i_mean_acc': itc_t2i_mean_acc,
        'itc_t2i_count': itc_t2i_count,
    }
    return ret


def compute_itm(model, batch, sim_dict=None):

    txt_ids = batch['text_ids']
    txt_mask = batch['text_mask']
    img = batch['image']
    bs = img.size(0)

    # positve pair
    output_pos = model.infer(batch, infer_mode='img-txt')

    # negative pair
    with torch.no_grad():
        if sim_dict is not None:
            sim_i2t = sim_dict['sim_i2t']
            sim_t2i = sim_dict['sim_t2i']
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-5
        else:
            weights_i2t = F.softmax(torch.randn([bs, bs]), dim=1) + 1e-5
            weights_t2i = F.softmax(torch.randn([bs, bs]), dim=1) + 1e-5

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    img_neg = []

    txt_ids_neg = []
    txt_mask_neg = []

    for b in range(bs):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        img_neg.append(img[neg_idx])
    img_neg = torch.stack(img_neg, dim=0)

    for b in range(bs):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        txt_ids_neg.append(txt_ids[neg_idx])
        txt_mask_neg.append(txt_mask[neg_idx])
    txt_ids_neg = torch.stack(txt_ids_neg, dim=0)
    txt_mask_neg = torch.stack(txt_mask_neg, dim=0)

    txt_ids_all = torch.cat([txt_ids, txt_ids_neg], dim=0)
    txt_mask_all = torch.cat([txt_mask, txt_mask_neg], dim=0)

    img_all = torch.cat([img_neg, img], dim=0)

    # forward negative pair
    itm_neg_batch = {
        'text_ids': txt_ids_all,
        'text_mask': txt_mask_all,
        'image': img_all,
    }
    output_neg = model.infer(itm_neg_batch, infer_mode='img-txt')

    cls_feat = torch.cat([output_pos['cls_feats'], output_neg['cls_feats']],
                         dim=0)
    itm_logits = model.itm_head(cls_feat)

    itm_labels = torch.cat([
        torch.ones(1 * bs, dtype=torch.long, device=itm_logits.device),
        torch.zeros(2 * bs, dtype=torch.long, device=itm_logits.device)
    ],
                           dim=0)

    itm_loss = F.cross_entropy(itm_logits, itm_labels)

    itm_mean_acc, itm_count = compute_accuracy(itm_logits, itm_labels)

    ret = {
        'itm_task_loss': itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
        "itm_mean_acc": itm_mean_acc,
        "itm_count": itm_count,
    }
    return ret


def compute_vqa(model, batch):
    """ train / val / infer Adapted """
    infer = model.infer(batch,
                        infer_mode='img-txt',
                        mask_txt=False,
                        mask_img=False)
    vqa_logits = model.vqa_classifier(infer["cls_feats"])

    ret = {"vqa_logits": vqa_logits, "vqa_count": vqa_logits.size(0)}

    vqa_targets = batch["vqa_targets"]

    if torch.sum(vqa_targets) > 0.0:
        vqa_loss = (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) *
            vqa_targets.shape[1])

        vqa_mean_score, vqa_count = compute_vqa_score(vqa_logits, vqa_targets)

        train_ret = {
            "vqa_task_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_mean_score": vqa_mean_score,
            "vqa_count": vqa_count,
        }
        ret.update(train_ret)

        if model.config.train.kl_alpha > 0.:
            infer_2 = model.infer(batch,
                                  infer_mode='img-txt',
                                  mask_txt=False,
                                  mask_img=False)
            vqa_logits_2 = model.vqa_classifier(infer_2["cls_feats"])
            vqa_loss_2 = (
                F.binary_cross_entropy_with_logits(vqa_logits_2, vqa_targets) *
                vqa_targets.shape[1])

            vqa_loss = (vqa_loss + vqa_loss_2) / 2.

            num_classes = vqa_targets.shape[1]
            p = torch.log_softmax(vqa_logits.view(-1, num_classes), dim=-1)
            p_tec = torch.softmax(vqa_logits.view(-1, num_classes), dim=-1)
            q = torch.log_softmax(vqa_logits_2.view(-1, num_classes), dim=-1)
            q_tec = torch.softmax(vqa_logits_2.view(-1, num_classes), dim=-1)
            kl = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
            r_kl = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

            kl_loss = (kl + r_kl) / 4 * model.config.train.kl_alpha

            kl_ret = {
                "vqa_task_loss": vqa_loss,
                "vqa_kl_task_loss": kl_loss,
            }
            ret.update(kl_ret)

    return ret


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


@torch.no_grad()
def dequeue_and_enqueue(model, image_feat, text_feat):

    if not model.training:
        return

    image_feats = concat_all_gather(image_feat)
    text_feats = concat_all_gather(text_feat)

    batch_size = image_feats.shape[0]

    ptr = int(model.queue_ptr)

    if ptr + batch_size <= model.q_size:
        model.img_queue[:, ptr:ptr + batch_size] = image_feats.T
        model.txt_queue[:, ptr:ptr + batch_size] = text_feats.T

    else:
        model.img_queue[:, ptr:] = image_feats.T[:, :model.q_size - ptr]
        model.txt_queue[:, ptr:] = text_feats.T[:, :model.q_size - ptr]
        model.img_queue[:, :ptr + batch_size -
                        model.q_size] = image_feats.T[:, model.q_size - ptr:]
        model.txt_queue[:, :ptr + batch_size -
                        model.q_size] = text_feats.T[:, model.q_size - ptr:]

    ptr = (ptr + batch_size) % model.q_size

    model.queue_ptr[0] = ptr


# ################### WIP ######################


def compute_mim(pl_module, batch):
    infer = pl_module.infer(batch, mask_txt=False, mask_img=True)
    mpp_logits = pl_module.mim_head(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(ret["mpp_logits"],
                                                      ret["mpp_labels"])
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


# ################### END ######################


def cost_matrix_cosine(x, y):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert (x.dim(), x.size(0), x.size(2)) == (y.dim(), y.size(0), y.size(2))
    return 1 - F.cosine_similarity(x.unsqueeze(2), y.unsqueeze(1), dim=3)


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool,
                     device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1,
                                                              keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype,
                       device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb,
                           img_emb,
                           txt_pad,
                           img_pad,
                           beta=0.5,
                           iteration=50,
                           k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) -
               txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) -
               img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta,
             iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len),
                            torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack([
            ti if itm_labels[i] == 1 else fi
            for i, (ti, fi) in enumerate(zip(bti, bfi))
        ])
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_txt=False, mask_img=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(
        ), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) -
                   txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
        img_len = (img_pad.size(1) -
                   img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
        T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
                 0.5, 50, 1)
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) +
                                                   dist_neg.size(0))

    itm_logits = pl_module.itm_head(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"],
                                                      ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_txt=False, mask_img=True)
    mpp_logits = pl_module.mpp_head(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(ret["mpp_logits"],
                                                      ret["mpp_labels"])
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_txt=False, mask_img=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_txt=False, mask_img=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_txt=False, mask_img=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(ret["imgcls_logits"],
                                                         ret["imgcls_labels"])
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(batch,
                             mask_txt=False,
                             mask_img=False,
                             image_token_type_idx=1)
    infer2 = pl_module.infer(batch,
                             mask_txt=False,
                             mask_img=False,
                             image_token_type_idx=2)

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(ret["nlvr2_logits"],
                                                            ret["nlvr2_labels"])
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [
            i for i, n in enumerate(batch["table_name"]) if "dev" in n
        ]
        test_batches = [
            i for i, n in enumerate(batch["table_name"]) if "test" in n
        ]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(F.cross_entropy(
                ret["nlvr2_logits"][dev_batches],
                ret["nlvr2_labels"][dev_batches]))
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches],
                ret["nlvr2_labels"][dev_batches])
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(F.cross_entropy(
                ret["nlvr2_logits"][test_batches],
                ret["nlvr2_labels"][test_batches]))
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches],
                ret["nlvr2_labels"][test_batches])
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks],
                           dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels],
                            dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h,
                                                   _w)

    infer = pl_module.infer({
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    })
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    ...


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json",
                  "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
