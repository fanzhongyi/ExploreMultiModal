import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


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

    def __init__(self, hidden_size, vocab_size, weight=None):
        super().__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class ITCHead(nn.Module):

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, out_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
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
