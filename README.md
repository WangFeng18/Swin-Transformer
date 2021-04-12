![Swin Transformer](./concept.png)

### Implementation of Swin Transformer

This repository contains the implementation of [Swin Transformer](https://arxiv.org/abs/2103.14030), and the training codes on ImageNet datasets. 

### Usage
Train on ImageNet:

Train Swin-T
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --model Swin_T \
--batch-size 192 --drop-path 0.2 --data-path ~/ILSVRC2012/ --output_dir /data/SwinTransformer_exp/SwinT/
```

Train Swin-S
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --model Swin_S \
--batch-size 192 --drop-path 0.3 --data-path ~/ILSVRC2012/ --output_dir /data/SwinTransformer_exp/SwinS/
```

Train Swin-B
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --model Swin_B \
--batch-size 192 --drop-path 0.5 --data-path ~/ILSVRC2012/ --output_dir /data/SwinTransformer_exp/SwinB/
```

### TODO
Training on ImageNet and give the detailed results.

### Reference
The training process involves many training and augmentation tricks, such as stochastic depth, mixup, cutmix and random erasing. I borrow large from Deit (https://github.com/facebookresearch/deit). 

### Citations

```bibtex
@misc{liu2021swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}, 
      author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
      year={2021},
      eprint={2103.14030},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
