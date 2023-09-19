## Environment Setup

Install dependencies by running:

```bash
conda create -n icm python=3.9
conda activate icm
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev

pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt
```

(Optional) install [xformers](https://github.com/facebookresearch/xformers) for efficient transformer implementation:
One could either install the pre-built version

```
pip install xformers==0.0.16
```

or build from latest source 

```bash
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
```

## TODO

# validation
现在只能在单个gpu log image, 指标计算是单卡还是多卡不清楚
calculate loss and log when bs > 1
多个validation dataloader

# dataset
区分完整前景和不完整前景（与边界是否相邻）
动物分20类 人不分
一共21类*2
"PPM", "AM2k_train", "AM2k_val", "RWP636", "P3M_val_np"

# 组合prompt
组合桌子和椅子对应的prompt

# 当前数据集大多聚焦于自动扣图 图片主题是显著的 不利于学习到语义上下文
solution:
1. 分割数据集
2.training free mask