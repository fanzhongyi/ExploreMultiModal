import math

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.patch_embed import PatchEmbed
from torchvision.transforms import transforms


def get_sinusoid_encoding_table_v2(n_position, d_hid):
    ''' Sinusoid position encoding table torch'''
    grid_x, grid_y = torch.meshgrid(torch.arange(n_position),
                                    torch.arange(d_hid))
    sinusoid_table = grid_x / torch.pow(
        10000, 2 * torch.div(grid_y, 2, rounding_mode='floor') / d_hid)
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
    return sinusoid_table.unsqueeze(0)


class PatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct the normalization for each patchs
        """
        super().__init__()
        self.gamma = nn.parameter.Parameter(torch.ones(hidden_size))
        self.beta = nn.parameter.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class PositionEmbed(nn.Module):
    def __init__(self, num_patches=196, d_model=768, num_tokens=0):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.num_tokens = num_tokens
        assert self.num_tokens >= 0, "num_tokens must be class token or no, so must 0 or 1"
        position = torch.arange(0, num_patches +
                                self.num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()
        pe = torch.zeros(num_patches + self.num_tokens, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def __call__(self):
        return self.pe


class MaskShuffle(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        ''' Get the masked embedding, maintain the mask position info. Note In-batch Samples make the same mask strategy.
            Args:
                x: [Batch, Length, Embed_dim]
                    x should contain position info and [CLS] token. eg. x = [CLS: embed] + pos_emb
                mask_ratio:
                    mask_ratio: 0.25, 0.50, 0.75, 0.95
            Return:
                visable_embedding (torch.Tensor):
                    [Batch, Length * (1 - mask_ratio), Embed_dim]
                visable_index (list):
                    [Length * (1 - mask_ratio) + 1(for [CLS]) ]
                    Token index 0 need assign to [CLS], which should keep visable.
                mask_index (list):
                    [Length * mask_ratio]
        '''
        length = x.size(1)
        shuffle_perm = torch.randperm(length - 1) + 1
        mask_length = round(length * self.mask_ratio)
        mask_index = shuffle_perm[:mask_length].tolist()
        visable_index = [0] + shuffle_perm[mask_length:].tolist()
        visable_embedding = x[:, visable_index, :]
        return visable_embedding, visable_index, mask_index


class UnMaskShuffle(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.input_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.kernel_size = patch_size

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=3,
                                      embed_dim=embed_dim,
                                      norm_layer=None)

        raw_inputs = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(
                torch.ones([3, img_size, img_size]) * 127. / 255).unsqueeze(0)
        self.register_buffer('raw_inputs', raw_inputs)

    def forward(self, x, visable_index=None):
        if visable_index:
            b, _, c = x.size()
            embedding = self.patch_embed(self.raw_inputs)  # [b = 1, l, c]
            embedding = embedding.expand(b, -1, -1)
            embedding = torch.cat((torch.zeros([b, 1, c]), embedding), dim=1)
            embedding[:, visable_index, :] = x
            return embedding
        return x


if __name__ == '__main__':
    a = torch.randperm(6)
    idx = torch.tensor([4, 3], dtype=torch.int64)
    exit()

    # NOTE: Test torch sin table
    #  from time import time
    #  start_t = time()
    #  res = get_sinusoid_encoding_table(1000, 1000)
    #  time1 = time() - start_t
    #
    #  start_t2 = time()
    #  res_v2 = get_sinusoid_encoding_table_v2(1000, 1000)
    #  time2 = time() - start_t2
    #
    #  print(res)
    #  print(res_v2)
    #  print(type(res), type(res_v2))
    #  print(torch.sum(res - res_v2))
    #  print(time1, time2)
    '''
    <class 'torch.Tensor'> <class 'torch.Tensor'>
    tensor(-0.0013)
    1.2690730094909668 0.004596233367919922
    '''
