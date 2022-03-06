import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class EstimatorCV(nn.Module):

    def __init__(self, hidden_size, class_num):
        super().__init__()
        self.class_num = class_num
        self.register_buffer("count", torch.zeros(class_num))
        self.register_buffer("mean", torch.zeros(class_num, hidden_size))
        self.register_buffer("cov", torch.zeros(class_num, hidden_size))

    @torch.no_grad()
    def forward(self, features, labels):
        C = self.class_num
        N, A = features.size()

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)  # (N, C, A)
        onehot = torch.zeros(N, C, device=features.device)  # (N, C)

        if labels.dim() == 2:
            onehot = labels.to(torch.bool).to(torch.long)
        else:
            onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)  # (N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)  # (N, C, A)

        Amount_CxA = NxCxA_onehot.sum(0)  # (C, A)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA  # (C, A)

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)
        weight_CV = sum_weight_CV.div(sum_weight_CV +
                                      self.count.view(C, 1).expand(C, A))

        weight_CV[weight_CV != weight_CV] = 0

        self.cov = (self.cov.mul(1 - weight_CV) + var_temp.mul(weight_CV) +
                    weight_CV.mul(1 - weight_CV).mul(
                        (self.mean - ave_CxA).pow(2))).detach()
        self.mean = (self.mean.mul(1 - weight_CV) +
                     ave_CxA.mul(weight_CV)).detach()
        self.count += onehot.sum(0)


class ISDAHead(nn.Module):

    def __init__(self, hidden_size, class_num):
        super().__init__()
        self.estimator = EstimatorCV(hidden_size, class_num)
        self.class_num = class_num

    def isda_aug(self, weight_m, features, labels, cv_matrix, ratio):

        C = self.class_num
        N, A = features.size()

        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))

        CV_temp = cv_matrix[labels]
        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)).sum(2)

        aug_result = 0.5 * sigma2
        return aug_result

    def forward(self, y, features, fc_weight, target, ratio):
        self.estimator(features.detach(), target)

        target = torch.max(target, 1)[1]
        isda_aug = self.isda_aug(fc_weight, features, target,
                                 self.estimator.cov.detach(), ratio)
        isda_aug_y = y + isda_aug
        return isda_aug_y


class MLMHead(nn.Module):

    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MIMHead(nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class ITCHead(nn.Module):

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.dense = nn.ModuleDict({
            'v': nn.Linear(hidden_size, out_size),
            'l': nn.Linear(hidden_size, out_size)
        })

    def forward(self, hidden_states, route=None):
        hidden_states = self.dense[route](hidden_states)
        hidden_states = nn.functional.normalize(hidden_states, dim=-1)
        return hidden_states


class ITMHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MPPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
