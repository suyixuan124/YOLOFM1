### 代码补充｜YOLOv5/v7/v8改进最新论文InceptionNeXt：当 Inception 遇到 ConvNeXt 系列，即插即用，小目标检测涨点必备改进

接博客内容

[YOLOv5/v7/v8改进最新论文InceptionNeXt：当 Inception 遇到 ConvNeXt 系列，即插即用，小目标检测涨点必备改进](https://blog.csdn.net/qq_38668236/article/details/129919463)

## 1. 改进的核心代码

```python
import torch
import torch.nn as nn

from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple


class Inception_DWConv2d(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class InceptionNeXtBlock(nn.Module):
    def __init__(
            self,dim_out,token_mixer=Inception_DWConv2d,norm_layer=nn.BatchNorm2d,mlp_layer=ConvMlp,
            mlp_ratio=4,act_layer=nn.GELU,ls_init_value=1e-6,drop_path=0.,):
        super().__init__()
        self.token_mixer = token_mixer(dim_out)
        self.norm = norm_layer(dim_out)
        self.mlp = mlp_layer(dim_out, int(mlp_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_input):
        shortcut = x_input
        x_input = self.token_mixer(x_input)
        x_input = self.norm(x_input)
        x_input = self.mlp(x_input)
        if self.gamma is not None:
            x_input = x_input.mul(self.gamma.reshape(1, -1, 1, 1))
        x_input = self.drop_path(x_input) + shortcut
        return x_input
```

## YOLOv5 + InceptionNeXt 代码改进

### 核心代码
在models文件夹下新增一个InceptionNeXt.py文件
```python
代码 如上目录《1. 改进的核心代码》👆
```

### 修改部分
在yolo.py中加入以下代码
```python
from models.inceptionnext import InceptionNeXtBlock
```
然后在 在`yolo.py`中配置
找到./models/yolo.py文件下里的`parse_model`函数

`for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):`内部
对应elif的位置 只需要增加 代码

参考代码
```python
        elif m in [InceptionNeXtBlock]:
            args = [ch[f], *args[1:]]
```

### YOLOv5-InceptionNeXt 网络配置文件
新增yolov5_InceptionNeXt.yaml
```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, InceptionNeXtBlock, []],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, InceptionNeXtBlock, []],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, InceptionNeXtBlock, []],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, InceptionNeXtBlock, []],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```
### 训练
运行命令：
python train.py --cfg yolov5_InceptionNeXt.yaml

-------

## YOLOv7 + InceptionNeXt 代码改进

### 核心代码
在models文件夹下新增一个InceptionNeXt.py文件
```python
代码 如上目录《1. 改进的核心代码》👆
```

### YOLOv7-InceptionNeXt 网络配置文件
新增yolov7_InceptionNeXt.yaml
```python
# parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 1.0  # layer channel iscyy multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0
   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4 
   [-1, 1, InceptionNeXtBlock, []], 
   [-1, 1, Conv, [256, 3, 1]], 
   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 16-P3/8
   [-1, 1, Conv, [128, 1, 1]],

   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],

   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],
   
   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],

   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],

   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],

   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],          
   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],
   [-1, 1, InceptionNeXtBlock, []],
   [-1, 1, Conv, [256, 3, 1]],
  ]

# yolov7 head
head:
  [[-1, 1, SPPCSPC, [512]],

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [31, 1, Conv, [256, 1, 1]],
   [[-1, -2], 1, Concat, [1]],
   [-1, 1, C3, [256]],

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [18, 1, Conv, [128, 1, 1]],
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, C3, [128]],

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, 44], 1, Concat, [1]],
   [-1, 1, C3, [256]], 
   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]], 
   [[-1, -3, 39], 1, Concat, [1]],

   [-1, 3, C3, [512]],

# head -----------------------------
   [49, 1, RepConv, [256, 3, 1]],
   [55, 1, RepConv, [512, 3, 1]],
   [61, 1, RepConv, [1024, 3, 1]],

   [[62,63,64], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
```

### 修改部分
在yolo.py中加入以下代码
```python
from models.inceptionnext import InceptionNeXtBlock
```
然后在 在`yolo.py`中配置
找到./models/yolo.py文件下里的`parse_model`函数

`for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):`内部
对应elif的位置 只需要增加 代码

参考代码
```python
        elif m in [InceptionNeXtBlock]:
            args = [ch[f], *args[1:]]
```

### 训练
运行命令：
python train.py --cfg yolov7_InceptionNeXt.yaml

-------

## YOLOv8 + InceptionNeXt 代码改进

### 核心代码

在ultralytics/nn文件夹下面，新增一个InceptionNeXt.py 文件，
核心代码
```python
代码 如上目录《1. 改进的核心代码》👆
```

### 修改部分
第二步：
在`ultralytics/nn/tasks.py`文件中

```python
from ultralytics.nn.InceptionNeXt import InceptionNeXtBlock
```


然后在 在`tasks.py`中配置
找到
```python
elif m is nn.BatchNorm2d:
   args = [ch[f]]
```
在这句上面加一个
```python
       elif m in (InceptionNeXtBlock):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c2, *args[1:]]
```

### YOLOv8 网络配置文件
新增一个 yolov8_InceptionNeXt.yaml
```python
# Ultralytics YOLO 🚀, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # scales module repeats
width_multiple: 0.50  # scales convolution channels

# YOLOv8.0s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 6, InceptionNeXtBlock, [128]],  # update
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, InceptionNeXtBlock, [256]],  # update
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0s head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```
### 训练