import torch
import torch.nn as nn
import numpy as np
import random
from einops import rearrange

from SwinTransformer import Swin_T, WMSA
from alignment.SwinTrans_msra import SwinTransformer, WindowAttention

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

setup_seed(0)

# a = Swin_T(10, drop_path_rate=0.0).cuda()
# print(a)
a = WMSA(input_dim=96, output_dim=96, head_dim=32, window_size=7, type='W').cuda()

setup_seed(0)
b = WindowAttention(dim=96, window_size=[7,7], num_heads=3).cuda()
print(b)
#b = Swin_T(10, drop_path_rate=0.0).cuda()

i = 0
# for c, d in zip(a.named_parameters(), b.named_parameters()):
#     print(c)
#     print(d)
#     i += 1

dummy_inputb = torch.rand(2*9,49,96).cuda()
dummy_inputa = rearrange(dummy_inputb, '(b W1 W2) (N1 N2) C -> b (W1 N1) (W2 N2) C', b=2,W1=3,N1=7)
output1 = a(dummy_inputa)
print(output1)
output2 = b(dummy_inputb)
print(output2)

