import torch
import torch.nn as nn
import numpy as np
import random
from einops import rearrange

from SwinTransformer import Swin_T
from alignment.SwinTrans_msra import SwinTransformer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)
input = torch.rand(2,3,224,224).cuda()

setup_seed(0)
a = Swin_T(num_classes=1000, drop_path_rate=0.2).cuda()

setup_seed(0)
b = SwinTransformer(num_classes=1000, drop_path_rate=0.2).cuda()

setup_seed(0)
output_a = a(input)

setup_seed(0)
output_b = b(input)

print(output_a)
print(output_b)
