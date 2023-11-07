### 代码补充｜YOLO系列全网首发改进最新：新颖特定任务检测头TSCODE｜(适用YOLOv5/v7)创新性Max，即插即用检测头，用于目标检测的特定任务上下文解耦头机制，助力YOLOv7目标检测器高效涨点！

接博客内容
## 二、改进YOLO + TSCODE核心模块代码
### 核心代码

在`yoloair库`的基础上，新增以下yaml文件
### YOLOv5-TSCODE 网络配置文件
新增yolov5_tscode.yaml
```python
# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors: 1  # number of anchors
loss: ComputeXLoss

# YOLOv5 backbone
backbone:
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 neck
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small) (80 80 256)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [17, 1, Conv, [256, 1, 1]],  # 24
   [20, 1, Conv, [256, 1, 1]],  # 25
   [23, 1, Conv, [256, 1, 1]],  # 26

   [[24, 4], 1, Concat, [1]], 
   [[25, 18], 1, Concat, [1]],  
   [[26, 21], 1, Concat, [1]],  

   [23, 1, nn.Upsample, [None, 2, 'nearest']],
   [-1, 1, DWConv, [512, 3, 1]],
   [20, 1, nn.Upsample, [None, 2, 'nearest']],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, 17], 1, Add, [1]],
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 1, Conv, [512, 3, 1]], # 36
   [[-1, 31, 20], 1, Add, [1]], 

   [23, 1, nn.Upsample, [None, 2, 'nearest']], 
   [-1, 1, DWConv, [512, 3, 1]],
   [[-1, 20], 1, Add, [1]],
   [-1, 1, DWConv, [512, 3, 2]],
   [-1, 1, DWConv, [1024, 3, 1]],
   [[-1, 23], 1, Add, [1]], # 43

   [27, 2, Conv, [256, 3, 1]],  # 44 cls0 (P3/8-small)
   [24, 2, Conv, [256, 3, 1]],  # 45 reg0 (P3/8-small)

   [28, 2, Conv, [256, 3, 1]],  # 46 cls1 (P4/16-medium)
   [37, 2, Conv, [256, 3, 1]],  # 47 reg1 (P4/16-medium)

   [29, 2, Conv, [256, 3, 1]],  # 48 cls2 (P5/32-large)
   [43, 2, Conv, [256, 3, 1]],  # 49 reg2 (P5/32-large)

    [[44, 45, 46, 47, 48, 49], 1, DetectX, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

在common.py中增加以下代码
```python
class Add(nn.Module):
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])
```

在yolo.py中加上
```python
elif m is Add:
     c2 = ch[f[0]]
     args = [c2]
```
如下图所示
![请添加图片描述](https://img-blog.csdnimg.cn/f66e92556185443696e74ac912a41f7c.png)
