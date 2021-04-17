import torch
import torch.nn as nn
import numpy as np
import random
from einops import rearrange

from SwinTransformer import Swin_T, WMSA
from alignment.SwinTrans_msra import SwinTransformer, WindowAttention, window_partition, window_reverse

def _fix_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.1)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    else:
        if hasattr(m, 'relative_position_params'):
            nn.init.constant_(m.relative_position_params, 0.1)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# a = Swin_T(10, drop_path_rate=0.0).cuda()
# print(a)
setup_seed(0)
input = torch.rand(2,21,21,96).cuda()

window_size = 7
setup_seed(0)
a = WMSA(input_dim=96, output_dim=96, head_dim=32, window_size=7, type='W').cuda()

setup_seed(0)
b = WindowAttention(dim=96, window_size=[7,7], num_heads=3).cuda()

output_a = a(input)
print(output_a)

x = input
# x = x.view(B, H, W, C)
shift_size = 0
# cyclic shift
if shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
else:
    shifted_x = x
# partition windows
x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
x_windows = x_windows.view(-1, window_size * window_size, 96)  # nW*B, window_size*window_size, C
# W-MSA/SW-MSA

H, W = 21,21
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size * window_size)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).cuda()

attn_windows = b(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

# merge windows
attn_windows = attn_windows.view(-1, window_size, window_size, 96)
shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C

# reverse cyclic shift
if shift_size > 0:
    x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
else:
    x = shifted_x
print(x)
#b = Swin_T(10, drop_path_rate=0.0).cuda()

# i = 0
# for c, d in zip(a.named_parameters(), b.named_parameters()):
#     print(c)
#     print(d)
#     i += 1